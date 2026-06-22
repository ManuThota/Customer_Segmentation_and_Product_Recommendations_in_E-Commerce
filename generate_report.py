import os
import sys
from datetime import datetime

# Add path for config imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import settings

def build_pdf():
    print("Generating project summary PDF report...")
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
    except ImportError:
        print("Reportlab not installed. Skipping PDF report generation.")
        return False
        
    pdf_path = settings.REPORT_PATH
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0F172A'), # Charcoal / Slate
        spaceAfter=8
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#64748B'), # Slate grey
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'H1Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1E3A8A'), # Navy
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'H2Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F766E'), # Teal
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'), # Dark Slate
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    story = []
    
    # ------------------ PAGE 1: TITLE & EXECUTIVE SUMMARY ------------------
    story.append(Paragraph("🛒 SHOPPER SPECTRUM", title_style))
    story.append(Paragraph("E-Commerce Customer Segmentation & Product Recommendation System", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Divider Line
    d_table = Table([[""]], colWidths=[500], rowHeights=[2])
    d_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1E3A8A')),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(d_table)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Executive Summary", h1_style))
    exec_summary_text = (
        "In modern retail, a one-size-fits-all marketing strategy is inefficient. "
        "Shopper Spectrum addresses this challenge by analyzing customer transaction patterns from online retail sales. "
        "The system executes two main tasks: (1) segmenting customers into behavioral groups based on Recency, Frequency, "
        "and Monetary (RFM) analysis using unsupervised K-Means clustering, and (2) generating personalized, item-based product "
        "recommendations using collaborative filtering. By implementing these methodologies, businesses can increase retention, "
        "optimize ad campaigns, and drive conversions through tailored product offerings."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    
    story.append(Paragraph("Project Objectives", h2_style))
    story.append(Paragraph("• <b>Understand Customer Behavior:</b> Clean, parse, and analyze raw retail transaction data.", bullet_style))
    story.append(Paragraph("• <b>Behavioral Segmentation:</b> Engineer RFM features, log-transform skewed data, and apply K-Means clustering to discover 4 distinct customer personas.", bullet_style))
    story.append(Paragraph("• <b>Collaborative Filtering:</b> Pivot customer-product transaction matrices and apply cosine similarity to provide similarity-based recommendations.", bullet_style))
    story.append(Paragraph("• <b>Interactive Deployment:</b> Deploy modules into a real-time Streamlit web app.", bullet_style))
    
    story.append(Spacer(1, 15))
    story.append(Paragraph("Dataset & Cleaning Pipeline", h1_style))
    dataset_text = (
        "The analysis is conducted on the <i>Online Retail Dataset</i>, containing transactions "
        "occurring between 2022 and 2023 for a UK-based online retail store. The dataset originally comprised 541,909 rows."
    )
    story.append(Paragraph(dataset_text, body_style))
    
    # Cleaning Table
    data_cleaning_steps = [
        ["Preprocessing Step", "Description", "Records Remaining"],
        ["Original Dataset", "Raw transaction database", "541,909"],
        ["Remove Null CustomerID", "Remove transactions lacking customer identification (24.9% of rows)", "406,829"],
        ["Exclude Cancelled Orders", "Remove returns/cancellations (InvoiceNo starting with 'C')", "397,924"],
        ["Filter Negative Quantities/Prices", "Filter zero or negative quantities/prices", "397,884"],
        ["Remove Duplicates", "Drop duplicate transaction entries", "392,692"]
    ]
    
    clean_table = Table(data_cleaning_steps, colWidths=[150, 230, 120])
    clean_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (2,0), (2,-1), 'CENTER')
    ]))
    story.append(clean_table)
    
    story.append(PageBreak())
    
    # ------------------ PAGE 2: EXPLORATORY DATA ANALYSIS & CLUSTERING ------------------
    story.append(Paragraph("Exploratory Data Analysis (EDA) & Plots", h1_style))
    story.append(Paragraph(
        "Key exploratory insights indicate that the United Kingdom generates over 90% of all transaction volume and revenue. "
        "The customer spending and order frequencies are heavily right-skewed, indicating that a minority of premium clients drive the majority of revenues.",
        body_style
    ))
    
    # Embed Country Analysis Plot if exists
    country_plot_path = os.path.join(settings.PLOT_DIR, 'country_analysis.png')
    if os.path.exists(country_plot_path):
        story.append(Spacer(1, 5))
        story.append(Image(country_plot_path, width=320, height=192))
        story.append(Paragraph("Figure 1: Top countries by transaction volume", ParagraphStyle('Cap', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#64748B'), alignment=1)))
    
    story.append(Paragraph("Customer Segmentation (RFM Analysis)", h1_style))
    story.append(Paragraph(
        "To group customers, we engineered three features: <b>Recency</b> (days since last purchase), "
        "<b>Frequency</b> (total number of transactions), and <b>Monetary</b> (total expenditures). "
        "Because Frequency and Monetary are highly skewed, we applied log transformation before scaling features with StandardScaler. "
        "A K-Means model was trained with K=4 clusters, which aligns with optimal elbow and silhouette metrics.",
        body_style
    ))
    
    # Embed Elbow Plot if exists
    elbow_plot_path = os.path.join(settings.PLOT_DIR, 'elbow_curve.png')
    if os.path.exists(elbow_plot_path):
        story.append(Spacer(1, 5))
        story.append(Image(elbow_plot_path, width=300, height=180))
        story.append(Paragraph("Figure 2: Elbow Method for selecting optimal cluster size", ParagraphStyle('Cap2', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#64748B'), alignment=1)))

    story.append(PageBreak())

    # ------------------ PAGE 3: SEGMENT CHARACTERISTICS & RECOMMENDATIONS ------------------
    story.append(Paragraph("Customer Segment Personas", h1_style))
    story.append(Paragraph(
        "The K-Means model successfully partitions the customer base into four actionable segments:",
        body_style
    ))
    
    # Segment Table
    segment_data = [
        ["Segment Persona", "Behavioral Characteristics", "Actionable Business Strategy"],
        ["High-Value", "Recent purchases, highly frequent buyers, and premium monetary spenders.", "VIP rewards, early product launches, custom loyalty account managers, referral bonuses."],
        ["Regular", "Consistent steady purchases with moderate spending. Represents the core base.", "Cross-sell recommendations, email updates, rewards milestone discounts."],
        ["Occasional", "Low frequency, low-to-medium spending. Active but occasionally engages.", "Flash sales, seasonal promotions, highlight specific item categories."],
        ["At-Risk", "Lapsed purchases (high recency), low spending, low frequency.", "Win-back coupon offers, feedback surveys, reactivating discount emails."]
    ]
    
    seg_table = Table(segment_data, colWidths=[110, 190, 200])
    seg_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F766E')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#99F6E4')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    story.append(seg_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("Product Recommendation System", h1_style))
    story.append(Paragraph(
        "For product recommendation, we implemented item-based collaborative filtering. "
        "We built a Customer-Product interaction matrix where cells represent total quantity of a product purchased by a customer. "
        "Cosine similarity is calculated between columns (product vector representations in customer space). "
        "The system recommends products that are purchased by similar clients, enabling real-time cross-selling.",
        body_style
    ))
    
    # Embed Cluster Plot if exists
    cluster_plot_path = os.path.join(settings.PLOT_DIR, 'cluster_visualization.png')
    if os.path.exists(cluster_plot_path):
        story.append(Spacer(1, 5))
        story.append(Image(cluster_plot_path, width=320, height=192))
        story.append(Paragraph("Figure 3: Log-scaled Scatterplot of Customer Clusters", ParagraphStyle('Cap3', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#64748B'), alignment=1)))
        
    story.append(Spacer(1, 20))
    # Footer metadata
    meta_text = f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Shopper Spectrum System"
    story.append(Paragraph(meta_text, ParagraphStyle('Meta', parent=styles['Normal'], fontSize=8, fontName='Helvetica-Oblique', textColor=colors.HexColor('#94A3B8'))))
    
    doc.build(story)
    print("Project summary report generated successfully!")
    return True

if __name__ == "__main__":
    build_pdf()
