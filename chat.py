import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
import numpy as np
from io import StringIO
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import calendar
import seaborn as sns
from scipy import stats

def load_and_process_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Read the CSV file with skipping first row
            df = pd.read_csv(uploaded_file, skiprows=1)
            df.columns = df.columns.str.strip()
            
            # Display raw data structure in expander
            with st.sidebar.expander("Data Preview & Columns"):
                st.write("Data Sample:")
                st.dataframe(df.head(2))
                st.write("Columns:", df.columns.tolist())
            
            # Convert date with better error handling
            date_column = next((col for col in df.columns if 'date' in col.lower()), None)
            if date_column:
                df[date_column] = pd.to_datetime(df[date_column], format='%d-%m-%Y')
                df['Month'] = df[date_column].dt.strftime('%b-%Y')
                df['Year'] = df[date_column].dt.year
                df['Quarter'] = df[date_column].dt.quarter
                df['Month_Num'] = df[date_column].dt.month
                df['Week'] = df[date_column].dt.isocalendar().week
                df['Day_of_Week'] = df[date_column].dt.day_name()
            
            # Clean monetary columns
            monetary_patterns = ['value', 'price', 'amount', 'rate', 'tax', 'cif', '$']
            monetary_cols = [col for col in df.columns 
                           if any(pattern in col.lower() for pattern in monetary_patterns)]
            
            for col in monetary_cols:
                df[col] = pd.to_numeric(df[col].astype(str)
                                      .str.replace('$', '')
                                      .str.replace(',', '')
                                      .str.strip(), errors='coerce')
            
            # Clean quantity columns
            qty_patterns = ['qty', 'quantity', 'amount', 'units']
            qty_cols = [col for col in df.columns 
                       if any(pattern in col.lower() for pattern in qty_patterns)]
            
            for col in qty_cols:
                df[col] = pd.to_numeric(df[col].astype(str)
                                      .str.replace(',', '')
                                      .str.strip(), errors='coerce')
            
            # Add computed metrics
            if 'Landed Value $' in df.columns and 'Standard Qty' in df.columns:
                df['Unit Value'] = df['Landed Value $'] / df['Standard Qty']
                
            if 'Tax $' in df.columns and 'Landed Value $' in df.columns:
                df['Tax Rate'] = (df['Tax $'] / df['Landed Value $']) * 100
            
            return df
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
    return None

def create_advanced_metrics(df):
    st.subheader("Advanced Metrics & KPIs")
    
    # Create three rows of metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Financial Metrics
        st.markdown("### 📊 Financial Metrics")
        total_value = df['Landed Value $'].sum()
        avg_shipment_value = df['Landed Value $'].mean()
        total_tax = df['Tax $'].sum() if 'Tax $' in df.columns else 0
        
        st.metric("Total Import Value", f"${total_value:,.2f}")
        st.metric("Avg Shipment Value", f"${avg_shipment_value:,.2f}")
        st.metric("Total Tax Paid", f"${total_tax:,.2f}")
    
    with col2:
        # Operational Metrics
        st.markdown("### 📦 Operational Metrics")
        total_shipments = len(df)
        unique_suppliers = df['Shipper Name'].nunique()
        unique_products = df['Product Description'].nunique()
        
        st.metric("Total Shipments", f"{total_shipments:,}")
        st.metric("Active Suppliers", f"{unique_suppliers:,}")
        st.metric("Product Portfolio", f"{unique_products:,}")
    
    with col3:
        # Performance Metrics
        st.markdown("### 📈 Performance Metrics")
        avg_tax_rate = df['Tax %'].mean() if 'Tax %' in df.columns else 0
        avg_unit_value = df['Unit Value'].mean() if 'Unit Value' in df.columns else 0
        
        st.metric("Avg Tax Rate", f"{avg_tax_rate:.2f}%")
        st.metric("Avg Unit Value", f"${avg_unit_value:.2f}")
        
def create_time_series_analysis(df):
    st.subheader("Advanced Time Series Analysis")
    
    # Create tabs for different time series views
    tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Seasonal Patterns", "Distribution Analysis"])
    
    with tab1:
        # Monthly trend with moving averages
        monthly_data = df.groupby('Month').agg({
            'Landed Value $': 'sum',
            'Standard Qty': 'sum',
            'Tax $': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=monthly_data['Month'],
            y=monthly_data['Landed Value $'],
            name='Actual Value',
            line=dict(color='blue')
        ))
        
        # Add 3-month moving average
        ma3 = monthly_data['Landed Value $'].rolling(window=3).mean()
        fig.add_trace(go.Scatter(
            x=monthly_data['Month'],
            y=ma3,
            name='3-Month MA',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title='Monthly Import Value with Moving Average',
            xaxis_title='Month',
            yaxis_title='Import Value ($)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Seasonal patterns
        if len(df['Month_Num'].unique()) >= 4:  # Check if enough data for seasonal analysis
            seasonal_data = df.groupby('Month_Num')['Landed Value $'].mean().reset_index()
            seasonal_data['Month'] = seasonal_data['Month_Num'].apply(lambda x: calendar.month_abbr[x])
            
            fig = px.line(seasonal_data, x='Month', y='Landed Value $',
                         title='Average Import Value by Month')
            st.plotly_chart(fig, use_container_width=True)
            
            # Day of week patterns
            dow_data = df.groupby('Day_of_Week')['Landed Value $'].mean().reset_index()
            fig = px.bar(dow_data, x='Day_of_Week', y='Landed Value $',
                        title='Average Import Value by Day of Week')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Distribution analysis
        st.markdown("### Value Distribution Analysis")
        
        # Create distribution plot
        fig = ff.create_distplot(
            [df['Landed Value $'].dropna()],
            ['Import Value'],
            bin_size=1000
        )
        fig.update_layout(title='Distribution of Import Values')
        st.plotly_chart(fig, use_container_width=True)
        
        # Add statistical summary
        st.markdown("### Statistical Summary")
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis'],
            'Value': [
                df['Landed Value $'].mean(),
                df['Landed Value $'].median(),
                df['Landed Value $'].std(),
                df['Landed Value $'].skew(),
                df['Landed Value $'].kurtosis()
            ]
        })
        st.dataframe(stats_df)

def create_product_analysis(df):
    st.subheader("Advanced Product Analysis")
    
    # Create tabs for different product analyses
    tab1, tab2, tab3 = st.tabs(["Portfolio Analysis", "Product Clustering", "Trend Analysis"])
    
    with tab1:
        # Product portfolio analysis
        product_metrics = df.groupby('Product Description').agg({
            'Landed Value $': ['sum', 'mean', 'count'],
            'Standard Qty': ['sum', 'mean'],
            'Tax $': 'sum'
        }).round(2)
        
        product_metrics.columns = ['Total Value', 'Avg Value', 'Frequency', 
                                 'Total Quantity', 'Avg Quantity', 'Total Tax']
        
        # Calculate additional metrics
        product_metrics['Value Share %'] = (product_metrics['Total Value'] / 
                                          product_metrics['Total Value'].sum() * 100)
        
        # Sort by value and calculate cumulative share for ABC analysis
        product_metrics = product_metrics.sort_values('Total Value', ascending=False)
        product_metrics['Cumulative Share %'] = product_metrics['Value Share %'].cumsum()
        
        # Assign ABC categories
        product_metrics['Category'] = pd.cut(
            product_metrics['Cumulative Share %'],
            bins=[0, 80, 95, 100],
            labels=['A', 'B', 'C']
        )
        
        st.markdown("### ABC Analysis")
        st.dataframe(product_metrics)
        
        # Create Pareto chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=product_metrics.index[:10],
            y=product_metrics['Value Share %'][:10],
            name='Value Share'
        ))
        fig.add_trace(go.Scatter(
            x=product_metrics.index[:10],
            y=product_metrics['Cumulative Share %'][:10],
            name='Cumulative Share',
            yaxis='y2'
        ))
        fig.update_layout(
            title='Product Pareto Analysis (Top 10)',
            yaxis2=dict(overlaying='y', side='right', range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Product clustering based on value and quantity
        if len(df) >= 3:  # Minimum required for clustering
            # Prepare data for clustering
            cluster_data = df.groupby('Product Description').agg({
                'Landed Value $': 'sum',
                'Standard Qty': 'sum'
            }).reset_index()
            
            # Normalize the features
            scaler = MinMaxScaler()
            features_normalized = scaler.fit_transform(
                cluster_data[['Landed Value $', 'Standard Qty']]
            )
            
            # Perform k-means clustering
            n_clusters = min(5, len(cluster_data))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_data['Cluster'] = kmeans.fit_predict(features_normalized)
            
            # Create scatter plot
            fig = px.scatter(
                cluster_data,
                x='Landed Value $',
                y='Standard Qty',
                color='Cluster',
                hover_data=['Product Description'],
                title='Product Clusters based on Value and Quantity'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Product trend analysis
        st.markdown("### Product Trend Analysis")
        
        # Select top products by value
        top_products = df.groupby('Product Description')['Landed Value $']\
                        .sum().nlargest(5).index
        
        # Create trend lines for top products
        product_trends = df[df['Product Description'].isin(top_products)]\
                          .pivot_table(
                              index='Month',
                              columns='Product Description',
                              values='Landed Value $',
                              aggfunc='sum'
                          )
        
        fig = px.line(product_trends, title='Top 5 Products Trend')
        st.plotly_chart(fig, use_container_width=True)

def create_supplier_analysis(df):
    st.subheader("Advanced Supplier Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Supplier Performance", "Geographic Analysis", "Relationship Analysis"])
    
    with tab1:
        # Comprehensive supplier metrics
        supplier_metrics = df.groupby('Shipper Name').agg({
            'Landed Value $': ['sum', 'mean', 'count'],
            'Standard Qty': ['sum', 'mean'],
            'Tax $': 'sum',
            'Product Description': 'nunique'
        }).round(2)
        
        supplier_metrics.columns = ['Total Value', 'Avg Value', 'Shipment Count',
                                  'Total Quantity', 'Avg Quantity', 'Total Tax',
                                  'Product Count']
        
        # Calculate additional metrics
        supplier_metrics['Value Share %'] = (supplier_metrics['Total Value'] / 
                                           supplier_metrics['Total Value'].sum() * 100)
        supplier_metrics['Avg Shipment Size'] = supplier_metrics['Total Quantity'] / \
                                               supplier_metrics['Shipment Count']
        
        st.markdown("### Supplier Performance Matrix")
        st.dataframe(supplier_metrics)
        
        # Create bubble chart
        fig = px.scatter(
            supplier_metrics.reset_index(),
            x='Shipment Count',
            y='Avg Value',
            size='Total Value',
            color='Product Count',
            hover_name='Shipper Name',
            title='Supplier Performance Matrix'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Geographic analysis
        if 'Country of Origin' in df.columns:
            geo_metrics = df.groupby('Country of Origin').agg({
                'Landed Value $': 'sum',
                'Standard Qty': 'sum',
                'Shipper Name': 'nunique'
            }).reset_index()
            
            fig = px.choropleth(
                geo_metrics,
                locations='Country of Origin',
                locationmode='country names',
                color='Landed Value $',
                hover_data=['Shipper Name', 'Standard Qty'],
                title='Import Value by Country'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Supplier relationship analysis
        st.markdown("### Supplier Relationship Patterns")
        
        # Create heatmap of supplier-product relationships
        pivot_table = pd.crosstab(df['Shipper Name'], df['Product Description'])
        fig = px.imshow(
            pivot_table,
            title='Supplier-Product Relationship Matrix',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)

def create_advanced_insights(df):
    st.subheader("Advanced Business Insights & Recommendations")
    
    # Create sections for different types of insights
    with st.expander("🎯 Strategic Insights", expanded=True):
        # Value Analysis
        total_value = df['Landed Value $'].sum()
        monthly_growth = df.groupby('Month')['Landed Value $'].sum().pct_change().mean() * 100
        
        # Supplier Concentration
        supplier_concentration = df.groupby('Shipper Name')['Landed Value $'].sum() / total_value * 100
        top_supplier_share = supplier_concentration.max()
        
        # Product Diversity
        product_concentration = df.groupby('Product Description')['Landed Value $'].sum() / total_value * 100
        top_product_share = product_concentration.max()
        
        insights = [
            f"📈 Import value shows {monthly_growth:.1f}% average monthly growth",
            f"🏭 Top supplier represents {top_supplier_share:.1f}% of total import value",
            f"📦 Top product category accounts for {top_product_share:.1f}% of imports"
        ]
        
        for insight in insights:
            st.markdown(insight)
            
    with st.expander("💰 Cost Optimization Opportunities"):
        # Tax Analysis
        avg_tax_rate = df['Tax %'].mean()
        tax_std = df['Tax %'].std()
        
        # Identify high-tax products
        high_tax_products = df[df['Tax %'] > (avg_tax_rate + tax_std)]\
                            .groupby('Product Description')['Tax %'].mean().sort_values(ascending=False)
        
        st.markdown("#### High Tax Products:")
        st.dataframe(high_tax_products.head().round(2))
        
        # Unit Cost Analysis
        if 'Unit Value' in df.columns:
            unit_value_analysis = df.groupby('Product Description')['Unit Value'].agg(['mean', 'std']).round(2)
            st.markdown("#### Unit Value Variation:")
            st.dataframe(unit_value_analysis.head())

    with st.expander("📊 Trend Analysis & Forecasting"):
        # Perform time series decomposition
        monthly_values = df.groupby('Month')['Landed Value $'].sum().reset_index()
        monthly_values.set_index('Month', inplace=True)
        
        try:
            decomposition = seasonal_decompose(monthly_values, period=min(len(monthly_values)-1, 12))
            
            fig = go.Figure()
            
            # Plot trend
            fig.add_trace(go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend.values,
                name='Trend',
                line=dict(color='blue')
            ))
            
            # Plot seasonal
            fig.add_trace(go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal.values,
                name='Seasonal',
                line=dict(color='green')
            ))
            
            fig.update_layout(title='Import Value Decomposition')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Not enough data points for trend decomposition")

    with st.expander("🤝 Supplier Relationship Insights"):
        # Supplier performance metrics
        supplier_metrics = df.groupby('Shipper Name').agg({
            'Landed Value $': ['sum', 'mean', 'std'],
            'Standard Qty': ['sum', 'mean'],
            'Record Id': 'count'
        }).round(2)
        
        supplier_metrics.columns = ['Total Value', 'Avg Value', 'Value Std',
                                  'Total Qty', 'Avg Qty', 'Shipment Count']
        
        # Calculate supplier reliability score
        supplier_metrics['Reliability Score'] = (
            (supplier_metrics['Shipment Count'] / supplier_metrics['Shipment Count'].max()) * 0.4 +
            (supplier_metrics['Total Value'] / supplier_metrics['Total Value'].max()) * 0.4 +
            (1 - (supplier_metrics['Value Std'] / supplier_metrics['Value Std'].max())) * 0.2
        ) * 100
        
        st.markdown("#### Top Performing Suppliers:")
        st.dataframe(supplier_metrics.sort_values('Reliability Score', ascending=False).head())

def create_advanced_filters(df):
    st.sidebar.header("Advanced Filters")
    
    # Date range with granularity selection
    date_granularity = st.sidebar.selectbox(
        "Date Granularity",
        ["Daily", "Weekly", "Monthly", "Quarterly"]
    )
    
    # Value range filter
    value_range = st.sidebar.slider(
        "Import Value Range ($)",
        min_value=float(df['Landed Value $'].min()),
        max_value=float(df['Landed Value $'].max()),
        value=(float(df['Landed Value $'].min()), float(df['Landed Value $'].max()))
    )
    
    # Multi-select filters
    selected_suppliers = st.sidebar.multiselect(
        "Select Suppliers",
        options=df['Shipper Name'].unique(),
        default=df['Shipper Name'].unique()
    )
    
    selected_products = st.sidebar.multiselect(
        "Select Products",
        options=df['Product Description'].unique(),
        default=df['Product Description'].unique()
    )
    
    # Advanced filters
    show_advanced = st.sidebar.checkbox("Show Advanced Filters")
    if show_advanced:
        selected_tax_range = st.sidebar.slider(
            "Tax Rate Range (%)",
            min_value=float(df['Tax %'].min()),
            max_value=float(df['Tax %'].max()),
            value=(float(df['Tax %'].min()), float(df['Tax %'].max()))
        )
        
        quantity_range = st.sidebar.slider(
            "Quantity Range",
            min_value=float(df['Standard Qty'].min()),
            max_value=float(df['Standard Qty'].max()),
            value=(float(df['Standard Qty'].min()), float(df['Standard Qty'].max()))
        )
    else:
        selected_tax_range = (float(df['Tax %'].min()), float(df['Tax %'].max()))
        quantity_range = (float(df['Standard Qty'].min()), float(df['Standard Qty'].max()))
    
    # Apply filters
    mask = (
        (df['Landed Value $'].between(value_range[0], value_range[1])) &
        (df['Shipper Name'].isin(selected_suppliers)) &
        (df['Product Description'].isin(selected_products)) &
        (df['Tax %'].between(selected_tax_range[0], selected_tax_range[1])) &
        (df['Standard Qty'].between(quantity_range[0], quantity_range[1]))
    )
    
    return df[mask], date_granularity

def create_download_section(df):
    st.sidebar.header("Export Options")
    
    export_format = st.sidebar.selectbox(
        "Select Export Format",
        ["CSV", "Excel", "JSON"]
    )
    
    if st.sidebar.button("Generate Export"):
        if export_format == "CSV":
            data = df.to_csv(index=False)
            mime = "text/csv"
            file_extension = "csv"
        elif export_format == "Excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            data = output.getvalue()
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            file_extension = "xlsx"
        else:
            data = df.to_json(orient='records')
            mime = "application/json"
            file_extension = "json"
        
        st.sidebar.download_button(
            label="Download Data",
            data=data,
            file_name=f"import_analysis.{file_extension}",
            mime=mime
        )

def main():
    st.set_page_config(page_title="Advanced Import Analysis Dashboard", layout="wide")
    
    st.title("Advanced Import Analysis Dashboard 2.0")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = load_and_process_data(uploaded_file)
        
        if df is not None:
            # Apply advanced filters
            filtered_df, date_granularity = create_advanced_filters(df)
            
            # Create download section
            create_download_section(filtered_df)
            
            # Display analysis sections
            create_advanced_metrics(filtered_df)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs([
                "Time Series Analysis",
                "Product Analysis",
                "Supplier Analysis",
                "Business Insights"
            ])
            
            with tab1:
                create_time_series_analysis(filtered_df)
            with tab2:
                create_product_analysis(filtered_df)
            with tab3:
                create_supplier_analysis(filtered_df)
            with tab4:
                create_advanced_insights(filtered_df)
    
    else:
        st.info("Please upload a CSV file to begin analysis")
        
        # Demo data option
        if st.button("Load Demo Data"):
            # Create sample data
            demo_df = create_demo_data()
            df = load_and_process_data(demo_df)
            if df is not None:
                filtered_df, date_granularity = create_advanced_filters(df)
                create_advanced_metrics(filtered_df)
                # Continue with rest of analysis...

if __name__ == "__main__":
    main()