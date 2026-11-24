import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Pharmacy Demand Insights", page_icon="💊", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.6rem}.hero{padding:1.5rem 1.8rem;border-radius:18px;background:linear-gradient(120deg,#064E3B,#0F9D78);color:white;margin-bottom:1rem}.hero h1{color:white;margin:0;font-size:2rem}.hero p{color:#D1FAE5;margin:.45rem 0 0}div[data-testid="stMetric"]{background:white;border:1px solid #E7ECF3;padding:14px;border-radius:14px}</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>💊 Pharmacy Sales & Demand Intelligence</h1><p>Explore medicine sales, product movement, and future demand estimates.</p></div>',unsafe_allow_html=True)
st.caption("Capstone prototype • Synthetic sample data • Forecasts are estimates, not purchase instructions")
DEFAULT="data/02_pharmacy_medicine_sales_synthetic.csv"
up=st.sidebar.file_uploader("Upload authorized sales CSV",type="csv")
try: df=pd.read_csv(up if up else DEFAULT)
except Exception as e: st.error(f"Could not load dataset: {e}"); st.stop()
required=["Transaction_ID","Transaction_Date","Product_Code","Product_Name","Product_Category","Quantity_Sold","Unit_Price","Sales_Amount","Stock_On_Hand_After_Sale","Restock_Recorded"]
missing=[c for c in required if c not in df.columns]
if missing: st.error("Missing columns: "+", ".join(missing)); st.stop()
df["Transaction_Date"]=pd.to_datetime(df.Transaction_Date,errors="coerce")
for c in ["Quantity_Sold","Unit_Price","Sales_Amount","Stock_On_Hand_After_Sale"]: df[c]=pd.to_numeric(df[c],errors="coerce")
with st.sidebar:
    st.header("Filters")
    products=st.multiselect("Products",sorted(df.Product_Name.dropna().unique()),default=sorted(df.Product_Name.dropna().unique()))
    cats=st.multiselect("Categories",sorted(df.Product_Category.dropna().unique()),default=sorted(df.Product_Category.dropna().unique()))
    dmin,dmax=df.Transaction_Date.min(),df.Transaction_Date.max()
    dr=st.date_input("Transaction dates",(dmin.date(),dmax.date())) if pd.notna(dmin) and pd.notna(dmax) else None
f=df[df.Product_Name.isin(products)&df.Product_Category.isin(cats)].copy()
if dr and len(dr)==2: f=f[(f.Transaction_Date.dt.date>=dr[0])&(f.Transaction_Date.dt.date<=dr[1])]
c1,c2,c3,c4=st.columns(4)
c1.metric("Sales records",f"{len(f):,}")
c2.metric("Units sold",f"{f.Quantity_Sold.sum():,.0f}")
c3.metric("Sales amount",f"₱{f.Sales_Amount.sum():,.2f}")
c4.metric("Products",f"{f.Product_Code.nunique():,}")
t1,t2,t3,t4=st.tabs(["Overview","Product movement","Demand forecast","Data"])
with t1:
    a,b=st.columns(2)
    monthly=f.dropna(subset=["Transaction_Date"]).assign(Month=lambda x:x.Transaction_Date.dt.to_period("M").astype(str)).groupby("Month").agg(Sales=("Sales_Amount","sum"),Units=("Quantity_Sold","sum")).reset_index()
    with a: st.plotly_chart(px.line(monthly,x="Month",y="Sales",markers=True,title="Monthly sales revenue",template="plotly_white"),use_container_width=True)
    with b:
        cat=f.groupby("Product_Category").Sales_Amount.sum().reset_index()
        st.plotly_chart(px.pie(cat,names="Product_Category",values="Sales_Amount",hole=.5,title="Revenue by category",template="plotly_white"),use_container_width=True)
    st.plotly_chart(px.bar(f.groupby("Product_Name").Quantity_Sold.sum().nlargest(10).reset_index(),x="Quantity_Sold",y="Product_Name",orientation="h",title="Top 10 products by units sold",template="plotly_white"),use_container_width=True)
with t2:
    stock=f.groupby(["Product_Code","Product_Name"],as_index=False).agg(Units_Sold=("Quantity_Sold","sum"),Latest_Stock=("Stock_On_Hand_After_Sale","last"),Revenue=("Sales_Amount","sum"))
    stock["Movement"]=np.where(stock.Units_Sold>=stock.Units_Sold.quantile(.75),"Higher movement","Regular/lower movement")
    st.dataframe(stock.sort_values("Units_Sold",ascending=False),use_container_width=True)
    st.info("Stock-on-hand is shown as recorded in the supplied dataset. Confirm current physical stock before making replenishment decisions.")
with t3:
    st.subheader("Product-level monthly demand forecast")
    choices=sorted(f.Product_Name.dropna().unique())
    if not choices: st.warning("No products match the selected filters.")
    else:
        prod=st.selectbox("Select medicine",choices)
        horizon=st.slider("Months to forecast",1,6,3)
        sub=f[f.Product_Name==prod].dropna(subset=["Transaction_Date"])
        monthly=sub.set_index("Transaction_Date").resample("MS").Quantity_Sold.sum().asfreq("MS",fill_value=0).reset_index()
        monthly.columns=["Month","Quantity"]
        if len(monthly)<6: st.warning("At least six monthly observations are recommended. Choose a product with more history.")
        else:
            n=max(1,int(len(monthly)*.2)); train=monthly.iloc[:-n]; test=monthly.iloc[-n:]
            Xtr=np.arange(len(train)).reshape(-1,1); Xte=np.arange(len(train),len(monthly)).reshape(-1,1)
            model=RandomForestRegressor(n_estimators=200,random_state=42) if len(train)>=12 else LinearRegression()
            model.fit(Xtr,train.Quantity); pred=np.maximum(0,model.predict(Xte))
            mae=mean_absolute_error(test.Quantity,pred); rmse=np.sqrt(mean_squared_error(test.Quantity,pred))
            m1,m2=st.columns(2); m1.metric("Holdout MAE",f"{mae:.2f}"); m2.metric("Holdout RMSE",f"{rmse:.2f}")
            # Refit on all history and extrapolate future months.
            full=RandomForestRegressor(n_estimators=200,random_state=42) if len(monthly)>=12 else LinearRegression()
            full.fit(np.arange(len(monthly)).reshape(-1,1),monthly.Quantity)
            future_idx=np.arange(len(monthly),len(monthly)+horizon).reshape(-1,1)
            future_dates=pd.date_range(monthly.Month.max()+pd.offsets.MonthBegin(1),periods=horizon,freq="MS")
            future=pd.DataFrame({"Month":future_dates,"Quantity":np.maximum(0,full.predict(future_idx)),"Type":"Forecast"})
            hist=monthly.copy(); hist["Type"]="Historical"
            chart=pd.concat([hist,future],ignore_index=True)
            st.plotly_chart(px.line(chart,x="Month",y="Quantity",color="Type",markers=True,title=f"Historical and estimated demand: {prod}",template="plotly_white"),use_container_width=True)
            st.dataframe(future,use_container_width=True)
            st.caption("Forecast method: Random Forest when at least 12 monthly points are available; otherwise Linear Regression. A simple time-index model is a baseline and may not capture seasonality.")
with t4:
    st.dataframe(f,use_container_width=True,height=420)
    st.download_button("Download filtered sales",f.to_csv(index=False).encode("utf-8"),"pharmacy_filtered_sales.csv","text/csv")
st.caption("Remove customer identifiers and sensitive prescription information from any real research dataset.")
