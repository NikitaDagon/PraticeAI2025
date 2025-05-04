from flask import Flask, request, send_file, jsonify, send_from_directory, make_response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
import uuid
import time
import pandas as pd
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app)

# Конфигурация
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
HISTORY_CSV = 'processing_history.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Инициализация истории
if not os.path.exists(HISTORY_CSV):
    pd.DataFrame(columns=[
        'id', 'filename', 'date',
        'frames_processed', 'objects_detected',
        'processing_time', 'output_path'
    ]).to_csv(HISTORY_CSV, index=False)

# Загрузка модели с калибровкой
model = YOLO("yolov8m.pt")  # Средняя модель для баланса точности/производительности
model.overrides['conf'] = 0.7  # Порог уверенности
model.overrides['iou'] = 0.45  # Подавление дубликатов
model.overrides['classes'] = [39]  # Только бутылки

# Глобальные переменные для прогресса
processing_status = {
    'current': 0,
    'total': 1,
    'message': '',
    'is_processing': False
}


@app.route('/')
def serve_index():
    return send_from_directory(STATIC_FOLDER, 'index.html')


@app.route('/api/process-video', methods=['POST'])
def process_video():
    global processing_status

    if processing_status['is_processing']:
        return jsonify({"error": "Сервер занят обработкой другого видео"}), 429

    processing_status['is_processing'] = True

    try:
        if 'video' not in request.files:
            return jsonify({"error": "Не предоставлен видеофайл"}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "Пустое имя файла"}), 400

        # Сохраняем временный файл
        file_id = str(uuid.uuid4())
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{file_id}.mp4")
        file.save(temp_path)

        # Получаем информацию о видео
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        processing_status['total'] = total_frames
        processing_status['current'] = 0
        processing_status['message'] = 'Начало обработки'

        # Обрабатываем видео
        output_path = os.path.join(UPLOAD_FOLDER, f"result_{file_id}.mp4")
        start_time = time.time()

        frames_processed, objects_detected = process_video_with_yolo(temp_path, output_path)

        processing_time = round(time.time() - start_time, 2)

        # Сохраняем в историю
        new_entry = {
            'id': file_id,
            'filename': file.filename,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'frames_processed': frames_processed,
            'objects_detected': objects_detected,
            'processing_time': processing_time,
            'output_path': output_path
        }

        history_df = pd.read_csv(HISTORY_CSV)
        history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
        history_df.to_csv(HISTORY_CSV, index=False)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        processing_status['is_processing'] = False
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


def process_video_with_yolo(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    total_objects = 0
    frames_processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Предварительная обработка кадра
        resized_frame = cv2.resize(frame, (640, 640))

        # 2. Детекция объектов
        results = model(resized_frame)

        # 3. Фильтрация результатов
        valid_objects = 0
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            area = width * height

            # Фильтр по размеру и пропорциям бутылки
            if (height / width > 1.2 and  # Пропорции бутылки
                    area > 500 and  # Минимальная площадь
                    height > 30):  # Минимальная высота
                valid_objects += 1

        # 4. Визуализация
        annotated_frame = results[0].plot(
            line_width=2,
            font_size=0.8,
            labels=False,
            conf=False
        )
        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))

        out.write(annotated_frame)
        total_objects += valid_objects
        frames_processed += 1

        processing_status['current'] = frames_processed
        processing_status['message'] = f'Обработано кадров: {frames_processed} | Найдено бутылок: {total_objects}'

    cap.release()
    out.release()

    return frames_processed, total_objects


@app.route('/api/progress')
def get_progress():
    return jsonify(processing_status)


@app.route('/api/history')
def get_history():
    history_df = pd.read_csv(HISTORY_CSV)
    return jsonify(history_df.to_dict('records'))


@app.route('/api/report/<report_type>/<file_id>')
def generate_report(report_type, file_id):
    history_df = pd.read_csv(HISTORY_CSV)
    record = history_df[history_df['id'] == file_id].iloc[0]

    if report_type == 'pdf':
        buffer = BytesIO()
        p = canvas.Canvas(buffer)

        p.drawString(100, 800, f"Отчет по обработке видео - {record['filename']}")
        p.drawString(100, 780, f"Дата: {record['date']}")
        p.drawString(100, 760, f"Обработано кадров: {record['frames_processed']}")
        p.drawString(100, 740, f"Обнаружено бутылок: {record['objects_detected']}")
        p.drawString(100, 720, f"Время обработки: {record['processing_time']} сек")

        p.showPage()
        p.save()

        buffer.seek(0)
        response = make_response(buffer.read())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=report_{file_id}.pdf'
        return response

    elif report_type == 'excel':
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            pd.DataFrame([record]).to_excel(writer, index=False)

        output.seek(0)
        response = make_response(output.read())
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        response.headers['Content-Disposition'] = f'attachment; filename=report_{file_id}.xlsx'
        return response

    return jsonify({"error": "Неверный тип отчета"}), 400


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)