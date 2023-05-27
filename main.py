from transformers import pipeline
from fastapi import FastAPI
import uvicorn

#引用各种模型
#语言是否积极
sentimentAnalysis = pipeline('sentiment-analysis')
#根据正文信息回答问题
question_answerer = pipeline('question-answering')
#文本生成
textGenerator = pipeline("text-generation")

app = FastAPI(title="抱抱脸接口测试")

@app.get("/sentiment", summary='情绪分析', tags=['文本相关'])
def qa(text: str = None):
    result = sentimentAnalysis(text)
    result = result[0]['label'] # 解析结果，只要需要的
    return {"code": 200, "result": result}

@app.get("/qa", summary='文本问答', tags=['文本相关'])
def qa(text: str = None, q_text: str = None):
    result = question_answerer({'question': q_text, 'context': text})
    result = result['answer'] # 解析结果，只要需要的
    return {"code": 200, "result": result}

@app.get("/tg", summary='文本生成', tags=['文本相关'])
def qa(text: str = None):
    result = textGenerator(text)
    return {"code": 200, "result": result}

if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8000)
