"""
make_exoplanet_guide_pdf.py
Convert EXOPLANET_GUIDE.html to EXOPLANET_GUIDE.pdf using fpdf2.

Run from the exoplanet/ subfolder:
    py -3 make_exoplanet_guide_pdf.py

Or from the repo root:
    py -3 whitenoise/exoplanet/make_exoplanet_guide_pdf.py

Requires: pip install fpdf2
Uses Windows Arial and Courier New fonts from C:/Windows/Fonts.
"""

import os
import re
import sys
from html.parser import HTMLParser

try:
    from fpdf import FPDF
except ImportError:
    print("fpdf2 not installed. Run: pip install fpdf2")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE  = os.path.dirname(os.path.abspath(__file__))
_HTML  = os.path.join(_HERE, 'EXOPLANET_GUIDE.html')
_OUT   = os.path.join(_HERE, 'EXOPLANET_GUIDE.pdf')

FONT_DIR = r'C:\Windows\Fonts'

# ── Unicode-to-ASCII map ──────────────────────────────────────────────────────
_UNICODE_MAP = {
    '’': "'",   '‘': "'",
    '“': '"',   '”': '"',
    '–': '-',   '—': '--',
    '·': '*',   'µ': 'mu',
    'μ': 'mu',  'β': 'beta',
    'α': 'alpha',
    '→': '->',  '»': '>>',
    '×': 'x',   '÷': '/',
    '²': '2',   '¹': '1',
    '³': '3',   '°': 'deg',
    '∞': 'inf',
    '≈': 'approx',
    '½': '1/2',
    '¼': '1/4',
    '≠': '!=',
    '≤': '<=',  '≥': '>=',
    '®': '(R)', '©': '(C)',
    'é': 'e',   'è': 'e',
    'à': 'a',   'á': 'a',
    'ü': 'u',   'ö': 'o',
    'ä': 'a',
    'Γ': 'Gamma',
    '−': '-',
    '±': '+/-',
    # HTML entities that slip through
    '&rarr;': '->',  '&larr;': '<-',
    '&middot;': '*', '&amp;': '&',
    '&lt;': '<',     '&gt;': '>',
    '&sup2;': '2',   '&asymp;': 'approx',
    '&le;': '<=',    '&ge;': '>=',
    '&mu;': 'mu',    '&beta;': 'beta',
    '&Gamma;': 'Gamma',
    '&sim;': '~',    '&ndash;': '-',
    '&mdash;': '--', '&lsquo;': "'",
    '&rsquo;': "'",  '&ldquo;': '"',
    '&rdquo;': '"',
}


def _sanitize(text: str) -> str:
    for k, v in _UNICODE_MAP.items():
        text = text.replace(k, v)
    return ''.join(c if ord(c) < 256 else '?' for c in text)


# ── Simple HTML parser ────────────────────────────────────────────────────────

class Block:
    def __init__(self, kind, content='', level=0, lang=''):
        self.kind    = kind    # heading|para|code|note|warn|tip|pre|hr|table|sig
        self.content = content
        self.level   = level
        self.lang    = lang
        self.rows    = []      # for table blocks


class HTMLGuideParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks   = []
        self._tag_stack = []
        self._buf     = ''
        self._in_pre  = False
        self._in_sig  = False
        self._in_table = False
        self._in_thead = False
        self._in_th   = False
        self._in_td   = False
        self._cur_row = []
        self._cur_table_rows = []
        self._in_callout = None   # 'note'|'warn'|'tip'|None
        self._in_style = False    # skip <style> block content
        self._in_head  = False    # skip <head> block content

    def _flush(self, kind='para', level=0, lang=''):
        text = self._buf.strip()
        if text:
            self.blocks.append(Block(kind, text, level, lang))
        self._buf = ''

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get('class', '')
        self._tag_stack.append(tag)

        if tag == 'style':
            self._in_style = True
            return
        if tag == 'head':
            self._in_head = True
            return
        if self._in_style or self._in_head:
            return

        if tag == 'pre':
            self._flush()
            self._in_pre = True
            self._buf = ''
        elif tag == 'span' and cls == 'sig':
            self._flush()
            self._in_sig = True
            self._buf = ''
        elif tag in ('h1', 'h2', 'h3'):
            self._flush()
            self._buf = ''
        elif tag == 'div' and cls in ('note', 'warn', 'tip'):
            self._flush()
            self._in_callout = cls
            self._buf = ''
        elif tag == 'div' and cls in ('pipeline',):
            self._flush()
            self._buf = ''
        elif tag == 'span' and cls in ('pipe-step', 'pipe-arrow', 'pipe-save', 'pipe-manual'):
            pass
        elif tag == 'hr':
            self._flush()
            self.blocks.append(Block('hr'))
        elif tag == 'table':
            self._flush()
            self._in_table = True
            self._cur_table_rows = []
        elif tag == 'thead':
            self._in_thead = True
        elif tag == 'tr':
            self._cur_row = []
        elif tag in ('th', 'td'):
            self._buf = ''
            self._in_th = tag == 'th'
            self._in_td = tag == 'td'
        elif tag == 'p' and self._in_callout:
            pass
        elif tag == 'p':
            self._flush()
            self._buf = ''
        elif tag == 'li':
            self._flush()
            self._buf = '- '

    def handle_endtag(self, tag):
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

        if tag == 'style':
            self._in_style = False
            self._buf = ''
            return
        if tag == 'head':
            self._in_head = False
            self._buf = ''
            return
        if self._in_style or self._in_head:
            return

        if tag == 'pre':
            self.blocks.append(Block('code', self._buf, lang=''))
            self._buf = ''
            self._in_pre = False
        elif tag == 'span' and self._in_sig:
            self.blocks.append(Block('sig', self._buf.strip()))
            self._buf = ''
            self._in_sig = False
        elif tag == 'h1':
            self.blocks.append(Block('heading', self._buf.strip(), level=1))
            self._buf = ''
        elif tag == 'h2':
            self.blocks.append(Block('heading', self._buf.strip(), level=2))
            self._buf = ''
        elif tag == 'h3':
            self.blocks.append(Block('heading', self._buf.strip(), level=3))
            self._buf = ''
        elif tag == 'div' and self._in_callout:
            text = self._buf.strip()
            if text:
                self.blocks.append(Block(self._in_callout, text))
            self._buf = ''
            self._in_callout = None
        elif tag in ('th', 'td'):
            cell = _sanitize(self._buf.strip())
            self._cur_row.append(cell)
            self._buf = ''
            self._in_th = False
            self._in_td = False
        elif tag == 'tr':
            if self._cur_row:
                self._cur_table_rows.append((self._in_thead, list(self._cur_row)))
            self._cur_row = []
        elif tag == 'thead':
            self._in_thead = False
        elif tag == 'table':
            blk = Block('table')
            blk.rows = self._cur_table_rows
            self.blocks.append(blk)
            self._in_table = False
            self._cur_table_rows = []
        elif tag == 'p':
            if self._in_callout:
                if self._buf.strip():
                    self._buf += ' '
            else:
                self._flush('para')
        elif tag == 'li':
            self._flush('list_item')

    def handle_data(self, data):
        if self._in_style or self._in_head:
            return
        if self._in_table and not (self._in_th or self._in_td):
            return
        self._buf += data

    def handle_entityref(self, name):
        entity = f'&{name};'
        self._buf += _UNICODE_MAP.get(entity, '')

    def handle_charref(self, name):
        try:
            if name.startswith('x'):
                c = chr(int(name[1:], 16))
            else:
                c = chr(int(name))
            self._buf += _sanitize(c)
        except Exception:
            pass


# ── PDF renderer ──────────────────────────────────────────────────────────────

NAVY  = (27, 58, 107)
GREY  = (244, 244, 244)
NOTE_BG = (238, 244, 251)
WARN_BG = (255, 248, 225)
TIP_BG  = (234, 250, 241)
WHITE = (255, 255, 255)
BLACK = (26, 26, 26)


class GuidePDF(FPDF):
    def setup_fonts(self):
        fd = FONT_DIR
        self.add_font('Arial',    '',   os.path.join(fd, 'arial.ttf'),    uni=True)
        self.add_font('Arial',    'B',  os.path.join(fd, 'arialbd.ttf'),  uni=True)
        self.add_font('Arial',    'I',  os.path.join(fd, 'ariali.ttf'),   uni=True)
        self.add_font('Arial',    'BI', os.path.join(fd, 'arialbi.ttf'),  uni=True)
        self.add_font('CourierNew', '', os.path.join(fd, 'cour.ttf'),     uni=True)
        self.add_font('CourierNew', 'B', os.path.join(fd, 'courbd.ttf'),  uni=True)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, 'whitenoise - Exoplanet SWNA Guide', align='L')
        self.ln(0.5)
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(*BLACK)

    def footer(self):
        self.set_y(-12)
        self.set_font('Arial', '', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 6, str(self.page_no()), align='C')

    def cover(self, title, subtitle, author):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, self.w, 38, 'F')
        self.set_font('Arial', 'B', 20)
        self.set_text_color(*WHITE)
        self.set_xy(14, 8)
        self.cell(0, 10, _sanitize(title), ln=True)
        self.set_font('Arial', 'I', 11)
        self.set_xy(14, 20)
        self.cell(0, 6, _sanitize(subtitle), ln=True)
        self.set_font('Arial', '', 9)
        self.set_xy(14, 29)
        self.cell(0, 6, _sanitize(author), ln=True)
        self.set_text_color(*BLACK)
        self.set_xy(self.l_margin, 46)

    def render_heading(self, text, level):
        text = _sanitize(text)
        self.ln(3)
        if level == 1:
            self.set_fill_color(*NAVY)
            self.set_text_color(*WHITE)
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, text, fill=True, ln=True)
        elif level == 2:
            self.ln(2)
            self.set_draw_color(*NAVY)
            self.set_line_width(0.8)
            self.set_font('Arial', 'B', 12)
            self.set_text_color(*NAVY)
            x0 = self.get_x()
            self.line(x0 - 1, self.get_y() + 1, x0 - 1, self.get_y() + 8)
            self.set_x(x0 + 3)
            self.cell(0, 9, text, ln=True)
            self.set_line_width(0.2)
        elif level == 3:
            self.set_font('Arial', 'B', 10)
            self.set_text_color(44, 62, 80)
            self.cell(0, 7, text, ln=True)
        self.set_text_color(*BLACK)
        self.ln(1)

    def render_para(self, text):
        text = _sanitize(text)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def render_code(self, text):
        text = _sanitize(text).rstrip()
        text = text.replace('\t', '    ')
        self.set_fill_color(*GREY)
        self.set_draw_color(221, 221, 221)
        self.set_font('CourierNew', '', 8)
        lines = text.split('\n')
        pad   = 3
        h_line = 4.5
        block_h = len(lines) * h_line + 2 * pad
        if self.get_y() + block_h > self.h - self.b_margin - 5:
            self.add_page()
        x0 = self.get_x()
        y0 = self.get_y()
        w  = self.w - self.l_margin - self.r_margin
        self.rect(x0, y0, w, block_h, 'FD')
        self.set_xy(x0 + pad, y0 + pad)
        for line in lines:
            self.set_x(x0 + pad)
            self.cell(0, h_line, line, ln=True)
        self.set_y(y0 + block_h + 3)

    def render_sig(self, text):
        text = _sanitize(text).strip()
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.set_font('CourierNew', '', 9)
        lines = text.split('\n')
        pad = 3
        h_line = 5.0
        block_h = len(lines) * h_line + 2 * pad
        if self.get_y() + block_h > self.h - self.b_margin - 5:
            self.add_page()
        x0 = self.get_x()
        y0 = self.get_y()
        w  = self.w - self.l_margin - self.r_margin
        self.rect(x0, y0, w, block_h, 'F')
        self.set_xy(x0 + pad, y0 + pad)
        for line in lines:
            self.set_x(x0 + pad)
            self.cell(0, h_line, line, ln=True)
        self.set_text_color(*BLACK)
        self.set_y(y0 + block_h + 3)

    def render_callout(self, text, kind):
        bg = {'note': NOTE_BG, 'warn': WARN_BG, 'tip': TIP_BG}.get(kind, GREY)
        accent = {'note': NAVY, 'warn': (230, 126, 34), 'tip': (39, 174, 96)}.get(kind, NAVY)
        text = _sanitize(text)
        self.set_font('Arial', '', 9.5)
        lines = self.multi_cell(0, 5, text, split_only=True)
        block_h = len(lines) * 5 + 8
        if self.get_y() + block_h > self.h - self.b_margin - 5:
            self.add_page()
        x0 = self.get_x()
        y0 = self.get_y()
        w  = self.w - self.l_margin - self.r_margin
        self.set_fill_color(*bg)
        self.rect(x0, y0, w, block_h, 'F')
        self.set_fill_color(*accent)
        self.rect(x0, y0, 1.5, block_h, 'F')
        self.set_xy(x0 + 5, y0 + 4)
        self.set_text_color(*BLACK)
        self.multi_cell(w - 6, 5, text)
        self.set_y(y0 + block_h + 3)

    def render_hr(self):
        self.ln(2)
        self.set_draw_color(221, 221, 221)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def render_list_item(self, text):
        text = _sanitize(text)
        if text.startswith('- '):
            text = text[2:]
        self.set_font('Arial', '', 10)
        x0 = self.l_margin
        self.set_x(x0 + 4)
        self.cell(4, 5.5, '-', ln=False)
        self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def render_table(self, rows):
        if not rows:
            return
        w = self.w - self.l_margin - self.r_margin
        n_cols = max(len(r[1]) for r in rows) if rows else 1
        col_w  = w / n_cols

        for is_header, cells in rows:
            if self.get_y() + 8 > self.h - self.b_margin:
                self.add_page()
            if is_header:
                self.set_fill_color(*NAVY)
                self.set_text_color(*WHITE)
                self.set_font('Arial', 'B', 8.5)
            else:
                self.set_fill_color(*WHITE)
                self.set_text_color(*BLACK)
                self.set_font('Arial', '', 9)

            for i, cell in enumerate(cells):
                cell = _sanitize(cell)
                fill = is_header
                self.cell(col_w, 7, cell[:40], border=1, fill=fill)
            self.ln()

        self.set_text_color(*BLACK)
        self.ln(3)


# ── Build PDF ─────────────────────────────────────────────────────────────────

def build_pdf():
    with open(_HTML, encoding='utf-8') as f:
        html = f.read()

    parser = HTMLGuideParser()
    parser.feed(html)
    blocks = parser.blocks

    pdf = GuidePDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(14, 14, 14)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.setup_fonts()
    pdf.add_page()

    # Cover from the first heading
    title_block = next((b for b in blocks if b.kind == 'heading' and b.level == 1), None)
    sub_block   = next((b for b in blocks if b.kind == 'para'), None)
    title  = title_block.content if title_block else 'Exoplanet SWNA Guide'
    sub    = 'SWNA on TESS Transit Light Curves | wn.exoplanet submodule'
    author = 'Framework: Bernido & Carpio-Bernido (2015) - World Scientific'
    pdf.cover(title, sub, author)

    skip_first_h1 = True
    for blk in blocks:
        if blk.kind == 'heading' and blk.level == 1 and skip_first_h1:
            skip_first_h1 = False
            continue

        if blk.kind == 'heading':
            pdf.render_heading(blk.content, blk.level)
        elif blk.kind == 'para':
            pdf.render_para(blk.content)
        elif blk.kind == 'list_item':
            pdf.render_list_item(blk.content)
        elif blk.kind == 'code':
            pdf.render_code(blk.content)
        elif blk.kind == 'sig':
            pdf.render_sig(blk.content)
        elif blk.kind in ('note', 'warn', 'tip'):
            pdf.render_callout(blk.content, blk.kind)
        elif blk.kind == 'hr':
            pdf.render_hr()
        elif blk.kind == 'table':
            pdf.render_table(blk.rows)

    pdf.output(_OUT)
    kb = os.path.getsize(_OUT) // 1024
    print(f'Saved: {_OUT}  ({kb} KB)')


if __name__ == '__main__':
    build_pdf()
