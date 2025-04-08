from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import easyocr
import io
import numpy as np

app = FastAPI()

# EasyOCR 리더 인스턴스 생성 (한국어와 영어를 동시에 인식)
# gpu=False로 설정하면 CPU 모드로 동작합니다.
reader = easyocr.Reader(['ko', 'en'], gpu=True)

@app.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):
    # 이미지 파일 여부 확인
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        # 업로드된 파일을 읽고 Pillow 이미지 객체로 변환
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        # EasyOCR은 numpy array 형식의 이미지를 사용하므로 변환
        np_img = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 중 오류 발생: {str(e)}")
    
    try:
        # EasyOCR를 이용해 텍스트 추출 (detail=0 옵션은 인식된 텍스트만 리스트로 반환)
        results = reader.readtext(np_img, detail=0)
        extracted_text = "\n".join(results)
        return JSONResponse(content={"extracted_text": extracted_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 추출 중 오류 발생: {str(e)}")

# uvicorn 실행 명령: uvicorn your_filename:app --reload
