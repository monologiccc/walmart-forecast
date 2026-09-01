import streamlit as st
import pandas as pd
import re

st.title("Мониторинг Walmart Sales Forecast API")

with open('logs/predictions.log') as f:
    lines = f.readlines()

records = []
pattern = r"store=(\d+) date=([\d-]+) prediction=([\d.]+)"
for line in lines:
    m = re.search(pattern, line)
    if m:
        records.append({
            'store': int(m.group(1)),
            'date': m.group(2),
            'prediction': float(m.group(3)),
        })

log_df = pd.DataFrame(records)

if not log_df.empty:
    st.metric("Всего запросов", len(log_df))
    st.line_chart(log_df.set_index('date')['prediction'])
    st.bar_chart(log_df['store'].value_counts())
else:
    st.info("Пока нет логов предсказаний")