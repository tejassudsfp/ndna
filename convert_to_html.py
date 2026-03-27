#!/usr/bin/env python3
"""
Convert paper_ndna.md to publication-quality HTML with MathJax support.
Protects math expressions from markdown processing, then restores them.
"""

import re
import markdown
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(SCRIPT_DIR, "paper_ndna.md")
HTML_PATH = os.path.join(SCRIPT_DIR, "paper_ndna.html")

def protect_math(text):
    """Replace math expressions with placeholders to prevent markdown from mangling them."""
    placeholders = {}
    counter = [0]

    def make_placeholder(match):
        key = f"MATHPLACEHOLDER{counter[0]}END"
        placeholders[key] = match.group(0)
        counter[0] += 1
        return key

    # Protect display math first ($$...$$), including multiline
    text = re.sub(r'\$\$(.+?)\$\$', make_placeholder, text, flags=re.DOTALL)

    # Protect inline math ($...$) - but not dollar signs used normally
    # Match $...$ where content doesn't start/end with space
    text = re.sub(r'\$([^\$\n]+?)\$', make_placeholder, text)

    return text, placeholders


def restore_math(html, placeholders):
    """Restore math expressions from placeholders."""
    for key, value in placeholders.items():
        html = html.replace(key, value)
    return html


def post_process_html(html):
    """Fix up the HTML after markdown conversion."""
    # Wrap tables in a div for overflow control
    html = re.sub(
        r'(<table>)',
        r'<div class="table-wrapper">\1',
        html
    )
    html = re.sub(
        r'(</table>)',
        r'\1</div>',
        html
    )
    return html


def build_html(body_html):
    """Wrap the converted body in a full HTML document with styling."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural DNA: A Compact Genome for Growing Network Architecture</title>

<!-- MathJax v3 -->
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    tags: 'ams',
    tagSide: 'right'
  }},
  svg: {{
    fontCache: 'global'
  }},
  startup: {{
    pageReady: () => {{
      return MathJax.startup.defaultPageReady().then(() => {{
        console.log('MathJax rendering complete');
        window.__mathjax_ready = true;
      }});
    }}
  }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>

<style>
/* Base styles */
* {{
  box-sizing: border-box;
}}

body {{
  font-family: Georgia, 'Times New Roman', Times, serif;
  font-size: 11pt;
  line-height: 1.6;
  color: #1a1a1a;
  max-width: 210mm;
  margin: 0 auto;
  padding: 2cm;
  background: #fff;
}}

/* Title page */
.title-block {{
  text-align: center;
  margin-bottom: 2em;
  padding-bottom: 1.5em;
}}

.title-block h1 {{
  font-size: 20pt;
  font-weight: bold;
  margin-bottom: 0.8em;
  line-height: 1.3;
  color: #000;
}}

.title-block .author {{
  font-size: 12pt;
  font-weight: bold;
  margin-bottom: 0.2em;
}}

.title-block .affiliation {{
  font-size: 11pt;
  font-style: italic;
  margin-bottom: 0.1em;
  color: #333;
}}

.title-block .contact {{
  font-size: 10pt;
  color: #444;
  margin-bottom: 0.1em;
}}

.title-block .contact a {{
  color: #2a5db0;
  text-decoration: none;
}}

hr.title-sep {{
  border: none;
  border-top: 2px solid #333;
  margin: 1.5em auto;
  width: 60%;
}}

/* Headings */
h1 {{
  font-size: 18pt;
  margin-top: 1.5em;
  margin-bottom: 0.5em;
  color: #000;
}}

h2 {{
  font-size: 14pt;
  margin-top: 1.8em;
  margin-bottom: 0.5em;
  color: #111;
  border-bottom: 1px solid #ccc;
  padding-bottom: 0.2em;
}}

h3 {{
  font-size: 12pt;
  margin-top: 1.2em;
  margin-bottom: 0.4em;
  color: #222;
}}

/* Paragraphs */
p {{
  margin-bottom: 0.8em;
  text-align: justify;
  hyphens: auto;
}}

/* Lists */
ol, ul {{
  margin-bottom: 0.8em;
  padding-left: 2em;
}}

li {{
  margin-bottom: 0.3em;
}}

/* Tables */
.table-wrapper {{
  width: 100%;
  overflow-x: auto;
  margin: 1em 0;
  page-break-inside: avoid;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85em;
  margin: 0.5em 0;
  page-break-inside: avoid;
}}

thead {{
  background-color: #f5f5f5;
}}

th {{
  font-weight: bold;
  text-align: left;
  padding: 6px 10px;
  border-bottom: 2px solid #333;
  border-top: 2px solid #333;
  white-space: nowrap;
}}

td {{
  padding: 5px 10px;
  border-bottom: 1px solid #ddd;
  vertical-align: top;
}}

tr:last-child td {{
  border-bottom: 2px solid #333;
}}

/* Bold in tables for highlighting best results */
td strong, th strong {{
  color: #000;
}}

/* Figures / Images */
.figure-block {{
  text-align: center;
  margin: 1.5em 0;
  page-break-inside: avoid;
}}

img {{
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.5em auto;
}}

/* Figure captions (italicized paragraphs inside figure blocks only) */
.figure-block p > em:only-child {{
  display: block;
  font-size: 0.9em;
  color: #444;
  text-align: left;
  margin-top: 0.3em;
}}

/* Code */
code {{
  font-family: 'Courier New', Courier, monospace;
  font-size: 0.9em;
  background-color: #f6f6f6;
  padding: 1px 4px;
  border-radius: 2px;
}}

pre {{
  background-color: #f6f6f6;
  padding: 0.8em;
  border-radius: 4px;
  overflow-x: auto;
  font-size: 0.85em;
  line-height: 1.4;
}}

pre code {{
  background: none;
  padding: 0;
}}

/* Links */
a {{
  color: #2a5db0;
  text-decoration: none;
}}

/* Math display */
.MathJax {{
  font-size: 100% !important;
}}

mjx-container[jax="SVG"][display="true"] {{
  margin: 0.8em 0 !important;
  overflow-x: auto;
}}

/* Blockquotes */
blockquote {{
  border-left: 3px solid #ccc;
  margin: 0.8em 0;
  padding: 0.3em 1em;
  color: #555;
  font-style: italic;
}}

/* References section */
.references p {{
  padding-left: 2em;
  text-indent: -2em;
  font-size: 0.92em;
  margin-bottom: 0.4em;
  line-height: 1.45;
}}

/* Print / PDF styles */
@page {{
  size: A4;
  margin: 2cm;
}}

@media print {{
  body {{
    padding: 0;
    font-size: 10.5pt;
    max-width: none;
  }}

  .title-block {{
    page-break-after: avoid;
  }}

  h2 {{
    page-break-before: auto;
    page-break-after: avoid;
  }}

  h3 {{
    page-break-after: avoid;
  }}

  /* Avoid breaking inside important elements */
  table, .table-wrapper {{
    page-break-inside: avoid;
  }}

  img {{
    page-break-inside: avoid;
    max-width: 90%;
  }}

  figure, .figure-block {{
    page-break-inside: avoid;
  }}

  /* Don't break right after headings */
  h1, h2, h3, h4 {{
    page-break-after: avoid;
  }}

  /* Major sections can have page breaks before them */
  .section-break {{
    page-break-before: always;
  }}

  /* Keep list items together */
  li {{
    page-break-inside: avoid;
  }}

  /* Keep paragraphs from orphans/widows */
  p {{
    orphans: 3;
    widows: 3;
  }}

  a {{
    color: #000;
    text-decoration: none;
  }}
}}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


def convert_title_block(html):
    """
    Convert the first h1 + author info into a centered title block.
    We build it directly from known content to avoid regex link issues.
    """
    # Find the title h1
    title_match = re.search(r'<h1>(.*?)</h1>', html)
    if not title_match:
        return html

    title = title_match.group(1)
    title_end = title_match.end()

    # Find the hr that ends the title block
    hr_match = re.search(r'<hr\s*/?\s*>', html[title_end:])
    if not hr_match:
        return html

    # Build title block directly from known content
    title_html = '''<div class="title-block">
  <h1>Neural DNA: A Compact Genome for Growing Network Architecture</h1>
  <div class="author">Tejas Parthasarathi Sudarshan</div>
  <div class="affiliation">Independent Researcher</div>
  <div class="affiliation">Chennai, India</div>
  <div class="contact"><a href="mailto:tejas@fandesk.ai">tejas@fandesk.ai</a> | <a href="https://tejassuds.com">tejassuds.com</a></div>
  <div class="contact">Code: <a href="https://github.com/tejassudsfp/ndna">github.com/tejassudsfp/ndna</a></div>
  <div class="contact">DOI: <a href="https://doi.org/10.5281/zenodo.19230474">10.5281/zenodo.19230474</a></div>
</div>
<hr class="title-sep" />
'''

    # Replace the original title block
    full_end = title_end + hr_match.end()
    html = html[:title_match.start()] + title_html + html[full_end:]

    return html


def add_section_breaks(html):
    """Add section-break class to ALL major h2 sections so each starts on a fresh page."""
    def add_break(match):
        heading = match.group(1)
        # Abstract stays on the title page, everything else gets a page break
        if heading.strip() == 'Abstract':
            return match.group(0)
        return f'<h2 class="section-break">{heading}</h2>'

    html = re.sub(r'<h2>(.*?)</h2>', add_break, html)
    return html


def wrap_figures(html):
    """Wrap img + caption in figure blocks for page break control."""
    # Pattern: <p><img ...></p> followed by <p><em>Figure...</em></p>
    html = re.sub(
        r'(<p><img[^>]+></p>\s*<p><em>(?:Figure|Fig)\s*\d+.*?</em></p>)',
        r'<div class="figure-block">\1</div>',
        html,
        flags=re.DOTALL
    )
    # Also wrap standalone images
    html = re.sub(
        r'(?<!<div class="figure-block">)(<p><img[^>]+></p>)(?!\s*<p><em>)',
        r'<div class="figure-block">\1</div>',
        html
    )
    return html


def wrap_references(html):
    """Wrap the References section content in a styled div."""
    # Find the References h2 and wrap everything until the next h2 or hr
    ref_match = re.search(r'(<h2[^>]*>References</h2>)', html)
    if not ref_match:
        return html

    ref_start = ref_match.end()
    # Find the next h2 or <hr after references
    next_section = re.search(r'(<h2|<hr)', html[ref_start:])
    if next_section:
        ref_end = ref_start + next_section.start()
    else:
        ref_end = len(html)

    ref_content = html[ref_start:ref_end]
    html = html[:ref_start] + '<div class="references">' + ref_content + '</div>' + html[ref_end:]
    return html


def main():
    # Read markdown
    with open(MD_PATH, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Step 1: Protect math expressions
    md_text, math_placeholders = protect_math(md_text)

    # Step 2: Convert markdown to HTML
    extensions = ['tables', 'fenced_code', 'smarty', 'sane_lists']
    body_html = markdown.markdown(md_text, extensions=extensions)

    # Step 3: Restore math expressions
    body_html = restore_math(body_html, math_placeholders)

    # Step 4: Post-process
    body_html = post_process_html(body_html)
    body_html = convert_title_block(body_html)
    body_html = add_section_breaks(body_html)
    body_html = wrap_references(body_html)
    body_html = wrap_figures(body_html)

    # Step 5: Build full HTML document
    full_html = build_html(body_html)

    # Step 6: Write output
    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"HTML written to {HTML_PATH}")
    print(f"Math expressions protected/restored: {len(math_placeholders)}")


if __name__ == '__main__':
    main()
