from flask import Flask, request, render_template, make_response, session, send_file
import joblib
import pandas as pd
import requests
import pdfkit
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from reportlab.lib.units import inch 
from flask import Flask, make_response, session, send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image
from reportlab.lib import colors
import io
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

app.secret_key = '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'  # Unique and secret key

API_TOKEN = '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'  # Replace with your actual API token

headers = {
    'Authorization': f'Bearer {API_TOKEN}',
}


def generate_forecast_explanation(forecast_df, product_name, product_uses):
    try:
        # Extract the row with the highest forecasted sales
        highest_sales_row = forecast_df.loc[forecast_df['Forecasted Sales'].idxmax()]
        # Extract the row with the lowest forecasted sales
        lowest_sales_row = forecast_df.loc[forecast_df['Forecasted Sales'].idxmin()]
        
        # Get the date range
        start_date = forecast_df['Date'].min().strftime('%B %Y')
        end_date = forecast_df['Date'].max().strftime('%B %Y')

        # Extract specific details
        highest_sales = highest_sales_row['Forecasted Sales']
        lowest_sales = lowest_sales_row['Forecasted Sales']
        highest_sales_date = highest_sales_row['Date'].strftime('%B %Y')
        lowest_sales_date = lowest_sales_row['Date'].strftime('%B %Y')

        # Calculate the lowest and highest possible sales values
        lowest_possible_value = forecast_df['Lower Bound'].min()
        highest_possible_value = forecast_df['Upper Bound'].max()

        # Calculate the number of months in the forecast
        num_months = len(forecast_df['Date'].unique())

        # Construct the explanatory text
        explanation = (
            f"This forecast covers a total of {num_months} months of projected sales for {product_name}, "
            f"ranging from {start_date} to {end_date}. {product_name} is commonly used for {product_uses}. "
            f"Each entry shows the forecasted sales along with the lower and upper bounds of the prediction. "
            f"The highest forecasted sales of {highest_sales:.2f} are projected for {highest_sales_date}. "
            f"This could be due to increased demand during this period, possibly driven by seasonal factors, "
            f"promotional activities, or a rise in health issues that {product_name} addresses. "
            f"In contrast, the lowest forecasted sales of {lowest_sales:.2f} are expected on {lowest_sales_date}, "
            f"which may be attributed to reduced demand, competition, or a lesser prevalence of conditions that the product treats during this time. "
            f"Additionally, the lowest possible sales value during this period could go as low as {lowest_possible_value:.2f}, "
            f"while the highest possible sales value could reach up to {highest_possible_value:.2f}. "
            f"The range of lower and upper bounds indicates the potential variability in sales figures for each date. "
            f"Understanding these forecasts helps in effective planning by highlighting periods with potential peaks and troughs in sales."
        )
        
    except Exception as e:
        return f"Error generating the forecast explanation: {e}"
    
    return explanation


def get_summary(forecast_df, product_name, product_uses):
    explanation = generate_forecast_explanation(forecast_df, product_name, product_uses)
    
    if "Error" in explanation:
        return explanation

    payload = {
        'inputs': explanation,
        'parameters': {'max_length': 400, 'min_length': 150},
    }
    response = requests.post(
        'https://api-inference.huggingface.co/models/facebook/bart-large-cnn',
        headers=headers,
        json=payload
    )
    
    try:
        summary = response.json()
        if isinstance(summary, list) and 'summary_text' in summary[0]:
            return summary[0]['summary_text']
        else:
            return "No summary could be generated. Please try again."
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error during summarization: {e}")
        return "There was an issue generating the summary. Please try again later."

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import plotly.graph_objs as go
import plotly.io as pio

def create_plot(forecast_df):
    # Create a figure
    fig = go.Figure()

    # Add the bar chart for forecasted sales
    fig.add_trace(
        go.Bar(
            x=forecast_df['Date'],
            y=forecast_df['Forecasted Sales'],
            name='Forecasted Sales (Bar)',
            marker=dict(color='lightblue'),
            opacity=0.6
        )
    )

    # Add the forecasted sales line with a solid line style
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecasted Sales'],
            mode='lines+markers',
            name='Forecasted Sales (Line)',
            line=dict(color='blue', width=3),
            marker=dict(size=6, symbol='circle')
        )
    )

    # Add the lower bound line with a dashed line style
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Lower Bound'],
            mode='lines+markers',
            name='Lower Bound',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6, symbol='triangle-down')
        )
    )

    # Add the upper bound line with a dotted line style
    fig.add_trace(
        go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Upper Bound'],
            mode='lines+markers',
            name='Upper Bound',
            line=dict(color='green', width=2, dash='dot'),
            marker=dict(size=6, symbol='triangle-up')
        )
    )

    # Update layout for better appearance
    fig.update_layout(
        title="Forecasted Sales with Confidence Intervals (Line and Bar Chart)",
        hovermode='x unified',
        yaxis_title="Sales",
        xaxis_title="Date",
        template='plotly_white',
        legend=dict(
            title="Legend",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        barmode='overlay'  # Overlay the bar chart with the lines
    )

    # Generate HTML for web interface
    graph_html = pio.to_html(fig, full_html=False)

    # Generate PNG for PDF
    image_bytes = fig.to_image(format="png")

    return graph_html, image_bytes

    





def get_recommendations(forecast_df):
    recommendation = []

    max_sales_row = forecast_df.loc[forecast_df['Forecasted Sales'].idxmax()]
    min_sales_row = forecast_df.loc[forecast_df['Forecasted Sales'].idxmin()]

    recommendation.append("Recommendations:")
    recommendation.append(f" - Focus on increasing production for {max_sales_row['Date'].strftime('%B %Y')} as it has the highest forecasted sales of {max_sales_row['Forecasted Sales']:.2f}.")
    recommendation.append(f" - Consider reducing inventory during {min_sales_row['Date'].strftime('%B %Y')}, which has the lowest forecasted sales of {min_sales_row['Forecasted Sales']:.2f}.")
    recommendation.append(" - Monitor the sales closely during the months with high upper bound values, as these periods may require additional resources to meet potential demand.")
    
    return "\n".join(recommendation)




@app.route('/', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        from_date = request.form['from_date']
        to_date = request.form['to_date']
        product = request.form['product']

        print(f"Selected product: {product}")

        # Define model paths and product uses based on the selected product
        if product == 'Zolpidem':
            rating_model_path = r'C:\Users\shari\OneDrive\Desktop\Demand web\model\sarimax_model_Zolpidem_rating_new.joblib'
            sales_model_path = r'C:\Users\shari\OneDrive\Desktop\Demand web\model\sarimax_model_Zolpidem.joblib'
            product_uses = "relieving allergy symptoms such as hay fever, sneezing, runny nose, and itchy eyes"
        # elif product == 'Diclofenac':
        #     rating_model_path = r'C:\Users\mahat\Downloads\Demand web\models\sarimax_model_Diclofenac_rating_new.joblib'
        #     sales_model_path = r'C:\Users\mahat\Downloads\Demand web\models\sarimax_model_Diclofenac_new (1).joblib'
        #     product_uses = "treating pain and inflammation, particularly in conditions like arthritis"
        # elif product == 'Aspirin':
        #     rating_model_path = r'C:\Users\mahat\Downloads\Demand web\models\rating_forecast_model.joblib'
        #     sales_model_path = r'C:\Users\mahat\Downloads\Demand web\models\sarimax_model_Aspirin_new.joblib'
        #     product_uses = "relieving pain, reducing fever, and acting as an anti-inflammatory"
        # elif product == 'Paracetamol':
        #     rating_model_path = r'C:\Users\mahat\Downloads\Demand web\models\sarimax_model_Paracetamol_rating_new.joblib'
        #     sales_model_path = r'C:\Users\mahat\Downloads\Demand web\models\sarimax_model_Paracetamol_NEW (1).joblib'
        #     product_uses = "relieving pain and reducing fever"
        # elif product == 'Zolpidem':
        #     rating_model_path = r'C:\Users\mahat\Downloads\Demand web\models\sarimax_model_Zolpidem_rating_new.joblib'
        #     sales_model_path = r'C:\Users\mahat\Downloads\Demand web\models\sarimax_model_Zolpidem.joblib'
        #     product_uses = "treating insomnia and helping with sleep initiation"
        # elif product == 'Benzodiazepines':
        #     rating_model_path = r'C:\Users\mahat\Downloads\Demand web\models\sarimax_model_Benzodiazepines_rating_new.joblib'
        #     sales_model_path = r'C:\Users\mahat\Downloads\Demand web\models\sarimax_model_Benzodiazepines.joblib'
        #     product_uses = "treating anxiety, seizures, and insomnia"
        # else:
        #     return "Product not found", 400

        # Load the models
        rating_model = joblib.load(rating_model_path)
        sales_model = joblib.load(sales_model_path)

        # Convert the date input into a date range
        forecast_start = pd.to_datetime(from_date, format='%Y-%m-%d')
        forecast_end = pd.to_datetime(to_date, format='%Y-%m-%d')
        forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq='M')

        # Forecast future ratings using the loaded rating model
        rating_forecast = rating_model.get_forecast(steps=len(forecast_index))
        forecast_rating_mean = rating_forecast.predicted_mean.round().astype(int).values.reshape(-1, 1)

        # Use the forecasted ratings as exogenous variables to forecast sales
        sales_forecast = sales_model.get_forecast(steps=len(forecast_index), exog=forecast_rating_mean)
        forecast_sales_mean = sales_forecast.predicted_mean
        forecast_sales_conf_int = sales_forecast.conf_int()

        # Prepare the sales forecast result for display
        forecast_sales_df = pd.DataFrame({
            'Date': forecast_index,
            'Forecasted Sales': forecast_sales_mean,
            'Lower Bound': forecast_sales_conf_int.iloc[:, 0].clip(lower=0),  # Convert negative values to zero
            'Upper Bound': forecast_sales_conf_int.iloc[:, 1]
        })

        # Round the forecasted sales and bounds to 2 decimal places
        forecast_sales_df['Forecasted Sales'] = forecast_sales_df['Forecasted Sales'].round(2)
        forecast_sales_df['Lower Bound'] = forecast_sales_df['Lower Bound'].round(2)
        forecast_sales_df['Upper Bound'] = forecast_sales_df['Upper Bound'].round(2)

        # Add Serial Number Column
        forecast_sales_df.reset_index(inplace=True)
        forecast_sales_df.index += 1  # Make serial number start from 1
        forecast_sales_df.index.name = 'Serial Number'
        forecast_sales_df.reset_index(inplace=True)
        
        # Identify the rows with the lowest and highest sales
        min_sales = forecast_sales_df['Forecasted Sales'].min()
        max_sales = forecast_sales_df['Forecasted Sales'].max()

        # Adding CSS class for highlighting only the rows with the highest and lowest sales
        forecast_sales_df['Highlight'] = forecast_sales_df['Forecasted Sales'].apply(
            lambda x: 'highlight-low' if x == min_sales else ('highlight-high' if x == max_sales else '')
        )


        # # Remove the 'Class' column
        # forecast_sales_df = forecast_sales_df.drop(columns=['Class'])

        # Format the dates to display only the date part
        # forecast_sales_df['Date'] = forecast_sales_df['Date'].dt.strftime('%Y-%m-%d')

        # Generate the explanation and summary from the BART model
        explanation = generate_forecast_explanation(forecast_sales_df, product, product_uses)
        summary = get_summary(forecast_sales_df, product, product_uses)

        graph_html, sales_plot_image = create_plot(forecast_sales_df)

        # Generate recommendations
        recommendations = get_recommendations(forecast_sales_df)


        # Store forecast data, explanation, summary, and recommendations in session
        session['forecast_sales_df'] = forecast_sales_df.to_json()
        session['explanation'] = explanation
        session['summary'] = summary
        session['recommendations'] = recommendations

        # Pass the data, explanation, summary, and recommendations to result.html
        return render_template(
            'result.html', 
            forecast_sales=forecast_sales_df.to_dict(orient='records'),
            explanation=explanation,
            summary=summary, 
            recommendations=recommendations,
            sales_plot=graph_html
        )

    return render_template('input.html')
    
@app.route('/download_pdf')
def download_pdf():
    # Retrieve forecast sales data, explanation, summary, and recommendations from session
    forecast_sales_json = session.get('forecast_sales_df', '{}')
    explanation = session.get('explanation', '')
    summary = session.get('summary', '')
    recommendations = session.get('recommendations', '')

    if not forecast_sales_json:
        return "No forecast data available for download."

    forecast_sales_df = pd.read_json(forecast_sales_json)

    # Generate graph image
    img_buffer = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.plot(forecast_sales_df['Date'], forecast_sales_df['Forecasted Sales'], label='Forecasted Sales')
    plt.fill_between(forecast_sales_df['Date'], forecast_sales_df['Lower Bound'], forecast_sales_df['Upper Bound'], color='lightgrey', alpha=0.5)
    plt.title('Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)  # Rewind the buffer to the beginning

    # Generate PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    content = []

    styles = getSampleStyleSheet()

    # Title
    content.append(Paragraph("Sales Forecast Results", styles['Title']))

    # Add forecast sales table
    data = [['Serial Number', 'Date', 'Forecasted Sales', 'Lower Bound', 'Upper Bound']]
    for index, row in forecast_sales_df.iterrows():
        data.append([
            row['Serial Number'],
            row['Date'].strftime('%B %Y'),
            f"{row['Forecasted Sales']:.2f}",
            f"{row['Lower Bound']:.2f}",
            f"{row['Upper Bound']:.2f}"
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    content.append(table)

    # Add explanation
    content.append(Paragraph("Explanation:", styles['Heading2']))
    content.append(Paragraph(explanation, styles['BodyText']))

    # Add summary
    content.append(Paragraph("Summary:", styles['Heading2']))
    content.append(Paragraph(summary, styles['BodyText']))

    # Add recommendations
    content.append(Paragraph("Recommendations:", styles['Heading2']))
    content.append(Paragraph(recommendations, styles['BodyText']))

    # Add graph image
    img = Image(img_buffer)
    img.drawHeight = 4*inch
    img.drawWidth = 6*inch
    content.append(img)

    doc.build(content)
    buffer.seek(0)

    # Send the PDF as a response
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=sales_forecast_results.pdf'
    return response

if __name__ == '__main__':
    app.run(debug=True)