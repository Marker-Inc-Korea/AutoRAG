#!/usr/bin/env python3
# /// script
# dependencies = ["reportlab==4.4.4", "python-docx==1.2.0", "openpyxl==3.1.5"]
# ///
"""Build the synthetic demo corpus for the AutoRAG Hermes plugin demo.

Everything is fictional. Writes only under the given root:
  <root>/docs                         files to index
  <root>/home                         sandbox HOME with Slack / Telegram / Discord crawler archives
Usage: uv run make-fixtures.py <root>
"""
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
if root.exists():
    shutil.rmtree(root)
docs = root / "docs"
home = root / "home"
docs.mkdir(parents=True)
home.mkdir(parents=True)

# ---------------------------------------------------------------- documents
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

styles = getSampleStyleSheet()
pdf = SimpleDocTemplate(str(docs / "Hanbit_Logistics_Contract_Renewal_2026.pdf"), pagesize=A4)
pdf.build([
    Paragraph("Master Logistics Services Agreement - Renewal 2026", styles["Title"]),
    Paragraph("Between Daon Foods Co. (Client) and Hanbit Logistics Ltd. (Provider).", styles["Normal"]),
    Spacer(1, 12),
    Paragraph("Article 4. Pricing", styles["Heading2"]),
    Paragraph("The unit price for cold-chain delivery is 10,800 KRW per box, fixed from "
              "December 1, 2026 through November 30, 2027.", styles["Normal"]),
    Paragraph("Article 7. Delivery", styles["Heading2"]),
    Paragraph("Deliveries are made on weekdays between 06:00 and 11:00. Saturday deliveries "
              "are not included in the unit price.", styles["Normal"]),
    Paragraph("Article 11. Termination", styles["Heading2"]),
    Paragraph("Either party may terminate this agreement by giving 30 days written notice.", styles["Normal"]),
])

from docx import Document
doc = Document()
doc.add_heading("Quote - Hanbit Logistics - v3", level=1)
doc.add_paragraph("Prepared for Daon Foods. Cold-chain delivery, Seoul metro area.")
doc.add_paragraph("Unit price: 10,800 KRW per box. Minimum 2,000 boxes per month.")
doc.add_paragraph("Valid until December 31, 2026.")
doc.save(docs / "Quote_Hanbit_v3_final.docx")
shutil.copy2(docs / "Quote_Hanbit_v3_final.docx", docs / "Quote_Hanbit_v3_final_FINAL_copy.docx")

(docs / "meeting-notes-2026-10.md").write_text(
    "# Vendor review - October 2026\n\n"
    "- Hanbit renewal is the biggest open item. Legal owns the contract text.\n"
    "- Packaging supplier moved to net-45 payment terms.\n"
    "- Next review: first week of December.\n",
    encoding="utf-8",
)

import openpyxl
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Boxes"
ws.append(["Month", "Boxes shipped", "Vendor"])
for row in [("2026-08", 2150, "Hanbit"), ("2026-09", 2310, "Hanbit"), ("2026-10", 2480, "Hanbit")]:
    ws.append(list(row))
wb.save(docs / "monthly-shipments.xlsx")

# ------------------------------------------------------------ Slack archive
env = dict(os.environ, HOME=str(home))
exp = root / "slack-export"
(exp / "general").mkdir(parents=True)
(exp / "users.json").write_text(
    '[{"id":"U01MINA","name":"mina","real_name":"Mina Park","profile":{"display_name":"mina"}},'
    '{"id":"U02JUNO","name":"juno","real_name":"Juno Kim","profile":{"display_name":"juno"}}]')
(exp / "channels.json").write_text(
    '[{"id":"C01GEN","name":"general","created":1760000000,"creator":"U01MINA","is_archived":false,'
    '"is_general":true,"members":["U01MINA","U02JUNO"],"topic":{"value":""},"purpose":{"value":"Company-wide"}}]')
(exp / "general" / "2026-11-03.json").write_text(
    '[{"type":"message","user":"U01MINA","text":"Heads up: the Hanbit Logistics renewal moves the unit price '
    'to 11,500 KRW per box, effective Dec 1. Procurement approved it on Friday.","ts":"1793000000.000100"}]')
subprocess.run(["slacrawl", "init", "-workspace", "T0DEMO01"], env=env, check=True, capture_output=True)
subprocess.run(["slacrawl", "--json", "import", "-workspace", "T0DEMO01", str(exp)],
               env=env, check=True, capture_output=True)
shutil.rmtree(exp)

# --------------------------------------------------------- Telegram archive
# telecrawl creates its own schema on first use; `status` on an empty HOME does exactly that.
subprocess.run(["telecrawl", "--json", "status"], env=env, check=True, capture_output=True)
tc = sqlite3.connect(home / ".telecrawl" / "telecrawl.db")
tc.execute("insert into chats(id,kind,name,message_count,last_message_at) values(1,'group','Hanbit x Ops',2,1793100000)")
for ev, mid, sj, sn, ts, text in [
    ("e1", "1", "S1", "Jaehyun Lee", 1793090000,
     "Confirmed with Hanbit by phone: delivery window for the Dec 1 renewal is Tue/Thu mornings only."),
    ("e2", "2", "S2", "Mina Park", 1793100000,
     "Please do not agree to Saturday deliveries, Hanbit charges a 15% weekend surcharge."),
]:
    cur = tc.execute(
        "insert into messages(event_id,source_pk,chat_jid,chat_name,msg_id,sender_jid,sender_name,ts,from_me,text)"
        " values(?,?,?,?,?,?,?,?,0,?)", (ev, int(mid), "1", "Hanbit x Ops", mid, sj, sn, ts, text))
    tc.execute("insert into messages_fts(rowid,text,chat,sender,media) values(?,?,?,?,?)",
               (cur.lastrowid, text, "Hanbit x Ops", sn, ""))
tc.commit()
tc.close()

# ---------------------------------------------------------- Discord archive
# Let discrawl create its own schema (it stamps a version), then add rows.
(home / "Library" / "Application Support" / "discord").mkdir(parents=True)
subprocess.run(["discrawl", "wiretap", "--json"], env=env, check=True, capture_output=True)
dc = sqlite3.connect(home / "Library" / "Application Support" / "discrawl" / "discrawl.db")
now = "2026-11-04T09:00:00Z"
dc.execute("insert into guilds(id,name,raw_json,updated_at) values('G1','Daon Team','{}',?)", (now,))
dc.execute("insert into channels(id,guild_id,kind,name,raw_json,updated_at,collection_scope)"
           " values('CH1','G1','text','vendors','{}',?,'guild')", (now,))
dc.execute("insert into members(guild_id,user_id,username,display_name,role_ids_json,raw_json,updated_at)"
           " values('G1','U9','sora','Sora Han','[]','{}',?)", (now,))
text = ("Hanbit Logistics sent the signed renewal PDF; legal wants the termination notice period "
        "set to 60 days, not 30.")
dc.execute("insert into messages(id,guild_id,channel_id,author_id,message_type,created_at,content,"
           "normalized_content,raw_json,updated_at) values('M1','G1','CH1','U9',0,'2026-11-04T08:30:00Z',?,?,'{}',?)",
           (text, text.lower(), now))
dc.execute("insert into message_fts(message_id,guild_id,channel_id,author_id,author_name,channel_name,content)"
           " values('M1','G1','CH1','U9','Sora Han','vendors',?)", (text,))
dc.commit()
dc.close()
print(f"demo corpus ready under {root}")
