from flask import Flask, request, jsonify, logging
import tensorflow as tf
import numpy as np

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
        # Head에서 전달된 중간 결과를 받아옴
        head_output = request.json['head_output']
        head_output = np.array(head_output)
        app.logger.debug(f"Received head output: {head_output.shape}")  # 로그 추가
        
        # Tail 모델 예측 수행
        prediction = tail_model.predict(head_output)
        app.logger.debug(f"Tail model prediction: {prediction}")  # 로그 추가
        
        # 가장 높은 확률을 가진 클래스 인덱스 추출
        predicted_index = np.argmax(prediction)
        predicted_label = LABELS[predicted_index]

        # 예측 결과 반환
        return jsonify({'label': predicted_label})
    except Exception as e:
        app.logger.debug(f"Prediction failed: {e}")  # 에러 로그 추가
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.logger.setLevel(10)
    app.run(host='0.0.0.0', port=8082)
