## tail 모델 추론 API
## tail 모델 추론 시간과 UI까지 결과 보내는 시간
## 총 2가지 측정
from flask import Flask, request, jsonify, logging
import tensorflow as tf
import numpy as np
import time


app = Flask(__name__)

# H5 파일로부터 모델 로드
MODEL_PATH = 'model/tail-alexnet-split-layer-2.h5'  # Tail 모델의 H5 파일
tail_model = tf.keras.models.load_model(MODEL_PATH)

# 레이블 정의
LABELS = ['Goreng', '뚝배기 스파게티', '빨래방', '아이스크림 할인점', '추억과 김밥', '커피나무']

# Tail 모델 예측 API 엔드포인트
@app.route('/tail_predict', methods=['POST'])
def tail_predict():
    try:
        # 시간 측정 시작 (전체 처리 시간)
        start_time_total = time.time()

        # Head에서 전달된 중간 결과를 받아옴
        head_output = request.json['head_output']
        head_output = np.array(head_output)
        app.logger.debug(f"Received head output: {head_output.shape}")

        # 시간 측정 시작 (Tail 모델 추론 시간)
        start_time_inference = time.time()

        # Tail 모델 예측 수행
        prediction = tail_model.predict(head_output)
        app.logger.debug(f"Tail model prediction: {prediction}")

        # Tail 모델 추론 시간 측정 종료 및 경과 시간 계산
        end_time_inference = time.time()
        elapsed_time_inference = end_time_inference - start_time_inference

        # 가장 높은 확률을 가진 클래스 인덱스 추출
        predicted_index = np.argmax(prediction)
        predicted_label = LABELS[predicted_index]

        # 전체 처리 시간 측정 종료 및 경과 시간 계산
        end_time_total = time.time()
        elapsed_time_total = end_time_total - start_time_total

        # 로그 출력
        app.logger.debug(f"Tail inference time: {elapsed_time_inference} seconds")
        app.logger.debug(f"Total processing time (including response preparation): {elapsed_time_total} seconds")

        # 예측 결과 반환
        return jsonify({'label': predicted_label})
    except Exception as e:
        app.logger.debug(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.logger.setLevel(10)
    app.run(host='0.0.0.0', port=8082)
