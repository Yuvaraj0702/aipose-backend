from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

def create_report(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Main Title
    main_title = Paragraph("<b>Your personalized self assessment</b>", styles['Title'])
    elements.append(main_title)
    elements.append(Spacer(1, 0.2 * inch))

    # Subtitle
    subtitle = Paragraph("Assessment Risk", styles['Title'])
    elements.append(subtitle)
    elements.append(Spacer(1, 0.2 * inch))

    # High Risk Section
    high_risk_title = Paragraph("<font color='white'>High health risk: Neck, Shoulders, Upper back</font>", styles['Heading2'])
    high_risk_title_back = colors.Color(0.8, 0.0, 0.0)  # Darker red background color
    elements.append(Table([[high_risk_title]], style=[('BACKGROUND', (0, 0), (0, 0), high_risk_title_back)]))
    elements.append(Spacer(1, 0.1 * inch))

    # Risk Details
    risks = [
        ("Cervical Spondylosis", "Neck", "Neck pain, stiffness, and sometimes numbness or weakness in the arms", "Leaning forward, Screen below eye level"),
        ("Thoracic Outlet Syndrome", "Neck and shoulders", "Neck pain, stiffness, and sometimes numbness or weakness in the arms", "Leaning forward, Screen below eye level"),
        ("Upper back strain", "Upper back", "Neck pain, stiffness, and sometimes numbness or weakness in the arms", "Leaning forward, Screen below eye level")
    ]

    for risk in risks:
        title, area, symptoms, causes = risk
        elements.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
        elements.append(Paragraph(f"Affected area: {area}", styles['Normal']))
        elements.append(Paragraph(f"Symptoms experienced: {symptoms}", styles['Normal']))
        elements.append(Paragraph(f"Likely caused by: {causes}", styles['Normal']))
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Spacer(1, 0.2 * inch))  # Space for image placeholder
        elements.append(Table([["[Placeholder for image]"]], style=[('BOX', (0, 0), (-1, -1), 0.25, colors.black)]))
        elements.append(Spacer(1, 0.2 * inch))

    # Medium Risk Section
    medium_risk_title = Paragraph("<font color='white'>Medium health risk: Lower back</font>", styles['Heading2'])
    medium_risk_title_back = colors.Color(0.8, 0.6, 0.0)  # Darker yellow background color
    elements.append(Table([[medium_risk_title]], style=[('BACKGROUND', (0, 0), (0, 0), medium_risk_title_back)]))
    elements.append(Spacer(1, 0.1 * inch))

    medium_risks = [
        ("Lumbar spondylosis", "Lower back", "Back pain and stiffness", "Leaning forward, Screen below eye level")
    ]

    for risk in medium_risks:
        title, area, symptoms, causes = risk
        elements.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
        elements.append(Paragraph(f"Affected area: {area}", styles['Normal']))
        elements.append(Paragraph(f"Symptoms experienced: {symptoms}", styles['Normal']))
        elements.append(Paragraph(f"Likely caused by: {causes}", styles['Normal']))
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Spacer(1, 0.2 * inch))  # Space for image placeholder
        elements.append(Table([["[Placeholder for image]"]], style=[('BOX', (0, 0), (-1, -1), 0.25, colors.black)]))
        elements.append(Spacer(1, 0.2 * inch))

    # Actions Section
    action_title = Paragraph("For your action", styles['Heading2'])
    elements.append(action_title)
    elements.append(Spacer(1, 0.1 * inch))

    actions = [
        ("Make immediate adjustments", [
                        "Sit back fully with back supported. Use a cushion or adjust lumbar pad.",
            "Place device close to your chair. Use a cushion or adjust lumbar pad.",
            "Sit back fully."
        ]),
        ("Self-help with DIY hacks and request products", [
            "Raise your screen to eye level"
        ]),
        ("Gradual habit changes", []),
        ("Request products to address medium risk issues", []),
        ("Educate self with recommended resources to continue improving well-being", [])
    ]

    for action in actions:
        action_header, action_steps = action
        elements.append(Paragraph(f"<b>{action_header}</b>", styles['Heading3']))
        for step in action_steps:
            elements.append(Paragraph(step, styles['Normal']))
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Spacer(1, 0.2 * inch))  # Space for image placeholder
        elements.append(Table([["[Placeholder for image]"]], style=[('BOX', (0, 0), (-1, -1), 0.25, colors.black)]))
        elements.append(Spacer(1, 0.2 * inch))

    # Build the document
    doc.build(elements)
