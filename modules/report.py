# 신고서 PDF 생성 모듈

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import datetime
import cv2
import tempfile
import os


def generate_report_pdf(violation, output_path):
    """Generate a PDF report for a violation."""
    c = canvas.Canvas(output_path, pagesize=letter)
    c.drawString(1*inch, 10*inch, "안전신문고 위반 신고서")
    c.drawString(1*inch, 9.5*inch, f"위반 유형: {violation.type_label}")
    c.drawString(1*inch, 9*inch, f"일시: {datetime.datetime.now()}")
    c.drawString(1*inch, 8.5*inch, f"장소: 데모 장소")
    plate = getattr(violation, 'plate', 'AI 인식 중...')
    c.drawString(1*inch, 8*inch, f"차량 번호: {plate}")
    c.drawString(1*inch, 7.5*inch, f"AI 신뢰도: {violation.confidence}%")
    
    # 첨부 영상 캡처
    if violation.snapshot is not None:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            cv2.imwrite(f.name, violation.snapshot)
            c.drawImage(f.name, 1*inch, 5*inch, width=4*inch, height=3*inch)
            os.unlink(f.name)
    
    c.save()