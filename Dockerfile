FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt

COPY app.py .
COPY student_result_model.pkl .
COPY metrics.json .

EXPOSE 5000

CMD ["python", "app.py"]
