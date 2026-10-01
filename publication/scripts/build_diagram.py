"""Compile the editable TikZ Figure 1; export vector SVG and a 300 dpi PNG."""
from pathlib import Path
import shutil, subprocess, tempfile
import pymupdf as fitz
O=Path(__file__).resolve().parents[1]
F=O/'figures'
with tempfile.TemporaryDirectory(prefix='ieee_access_figure1_') as folder:
    result=subprocess.run(['pdflatex','--interaction=nonstopmode','--halt-on-error','--disable-write18',
        '--output-directory='+folder,'fig01_design.tex'],cwd=F,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
    if result.returncode:
        raise RuntimeError(result.stdout)
    if 'Overfull' in result.stdout:
        raise RuntimeError('Figure contains overflowing text: '+result.stdout)
    shutil.copy2(Path(folder)/'fig01_design.pdf',F/'fig01_design.pdf')
with fitz.open(F/'fig01_design.pdf') as doc:
    doc[0].get_pixmap(dpi=300).save(F/'fig01_design.png')
    (F/'fig01_design.svg').write_text(doc[0].get_svg_image(text_as_path=False),encoding='utf-8')
print('Figure 1: TikZ PDF, vector SVG and 300 dpi PNG generated.')
