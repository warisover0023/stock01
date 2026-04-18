import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
from scipy.linalg import eigh
import warnings
import smtplib
from email.mime.text import MIMEText
import os

# 警告を非表示にする
warnings.filterwarnings('ignore')

# --- 設定パラメータ ---
L = 60          
LAMBDA = 0.90   
K = 3           

JP_SECTORS = {
    '1617.T': "食品", '1618.T': "エネルギー", '1619.T': "建設・資材", '1620.T': "素材・化学",
    '1621.T': "医薬品", '1622.T': "自動車", '1623.T': "鉄鋼・非鉄", '1624.T': "機械",
    '1625.T': "電機・精密", '1626.T': "情報通信", '1627.T': "電力・ガス", '1628.T': "運輸・物流",
    '1629.T': "商社・卸売", '1630.T': "小売", '1631.T': "銀行", '1632.T': "金融[除く銀行]", '1633.T': "不動産"
}
US_SECTORS = {'XLB': "素材", 'XLC': "通信", 'XLE': "エネルギー", 'XLF': "金融", 'XLI': "資本財", 'XLK': "情報技術", 'XLP': "生活必需品", 'XLRE': "不動産", 'XLU': "公益", 'XLV': "ヘルスケア", 'XLY': "一般消費財"}
cyclical_names = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLY', '1618.T', '1619.T', '1620.T', '1622.T', '1623.T', '1624.T', '1625.T', '1629.T', '1631.T', '1632.T']

def send_email(content):
    mail_addr = os.environ.get('MAIL_ADDRESS')
    mail_pass = os.environ.get('MAIL_PASSWORD')
    if not mail_addr or not mail_pass:
        print("Mail settings missing")
        return
    msg = MIMEText(content)
    msg['Subject'] = '【朝刊】株価予測シグナル'
    msg['From'] = mail_addr
    msg['To'] = mail_addr
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(mail_addr, mail_pass)
        smtp.send_message(msg)

def get_refined_signal():
    all_tickers = list(US_SECTORS.keys()) + list(JP_SECTORS.keys())
    data = yf.download(all_tickers, period='120d', progress=False)
    df_close = data['Close'].ffill()
    df_open = data['Open'].ffill()
    us_tickers = [t for t in US_SECTORS.keys() if t in df_close.columns]
    jp_tickers = [t for t in JP_SECTORS.keys() if t in df_close.columns]
    us_ret = np.log(df_close[us_tickers] / df_close[us_tickers].shift(1)).dropna()
    jp_ret = np.log(df_close[jp_tickers] / df_open[jp_tickers]).dropna()
    aligned_us_ret = us_ret.reindex(jp_ret.index - pd.Timedelta(days=1), method='pad')
    aligned_us_ret.index = jp_ret.index
    combined_ret = pd.concat([aligned_us_ret, jp_ret], axis=1).dropna()
    if len(combined_ret) < L: return None
    recent_rets = combined_ret.tail(L)
    z_rets = (recent_rets - recent_rets.mean()) / recent_rets.std()
    lw = LedoitWolf().fit(z_rets)
    Ct = lw.covariance_
    n = len(us_tickers) + len(jp_tickers)
    v1 = np.ones(n) / np.sqrt(n)
    v2 = np.zeros(n)
    v2[:len(us_tickers)] = 1
    v2[len(us_tickers):] = -1
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.array([1 if t in cyclical_names else -1 for t in (us_tickers + jp_tickers)])
    v3 = v3 / np.linalg.norm(v3)
    V0 = np.column_stack([v1, v2, v3])
    V0, _ = np.linalg.qr(V0)
    C0 = V0 @ V0.T
    Ct_reg = (1 - LAMBDA) * Ct + LAMBDA * C0
    vals, vecs = eigh(Ct_reg, subset_by_index=(n-K, n-1))
    Vt_k = vecs[:, ::-1]
    Vu = Vt_k[:len(us_tickers), :]
    Vj = Vt_k[len(us_tickers):, :]
    B = Vj @ Vu.T
    latest_us_ret = z_rets[us_tickers].iloc[-1].values
    predicted_jp_z = B @ latest_us_ret
    return pd.Series(predicted_jp_z, index=jp_tickers).sort_values(ascending=False)

if __name__ == "__main__":
    signal = get_refined_signal()
    if signal is not None:
        report = "■ PCA-sub リードラグ戦略 予測シグナル\n" + "="*30 + "\n"
        report += "【買い推奨 (Top 5)】\n"
        for ticker, score in signal.head(5).items():
            report += f" ・ {ticker} | {JP_SECTORS[ticker]} ({score:+.4f})\n"
        report += "\n【空売り推奨 (Bottom 5)】\n"
        for ticker, score in signal.tail(5)[::-1].items():
            report += f" ・ {ticker} | {JP_SECTORS[ticker]} ({score:+.4f})\n"
        send_email(report)
        print("Done")
