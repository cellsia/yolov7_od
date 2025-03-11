from pathlib import Path
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.colors import HexColor

def create_signature_box():
    data = [
        ["Written by: ", "Approved by: "],
        ["Signature and date: ", "Signature and date: "],
        ["Name and surname: ", "Name and surname: "]
    ]
    page_width = letter[0]
    left_margin, right_margin = 80, 80
    usable_width = page_width - left_margin - right_margin
    col_widths = [usable_width * 0.5, usable_width * 0.5]
    table = Table(data, colWidths=col_widths)

    table_style = TableStyle([
        ('ALIGN', (0, 0), (-1, 0), 'LEFT'),
        ('ALIGN', (0, 1), (-1, 1), 'RIGHT'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black), 
        ('TOPPADDING', (0, 1), (-1, 1), 40), 
        ('BOTTOMPADDING', (0, 2), (-1, 2), 30),
        ('BOX', (0, 0), (-1, -1), 1, colors.black), 
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    table.setStyle(table_style)
    return table

from reportlab.lib.colors import HexColor

def rgb_to_hex(color):
    """Convierte un color RGB en formato (R, G, B) a un string hexadecimal '#RRGGBB'."""
    return HexColor('#{:02x}{:02x}{:02x}'.format(*color))

def create_legend(class_colors):
    """
    Crea una leyenda horizontal con recuadros de colores y nombres de clase sin normalizar.
    """
    legend_data = []
    current_row = []

    for idx, (cls, color) in enumerate(class_colors.items()):
        try:
            hex_color = rgb_to_hex(color)  # Convertir RGB a HEX
        except Exception as e:
            print(f"Error con el color de la clase {cls}: {color} - {e}")
            hex_color = HexColor("#000000")  # Color negro por defecto en caso de error

        # Crear el recuadro de color
        color_box = Table(
            [[" "]],  # Espacio vacío dentro del recuadro
            colWidths=10, rowHeights=10
        )
        color_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), hex_color),  # Usamos HEX en vez de normalizar
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ]))

        # Añadir recuadro y nombre a la fila actual
        current_row.append(color_box)
        current_row.append(Paragraph(str(cls), getSampleStyleSheet()['Normal']))

        # Si alcanzamos un número par de elementos en una fila o es el último elemento, agregamos la fila
        if len(current_row) >= 15 or idx == len(class_colors) - 1:
            legend_data.append(current_row)
            current_row = []

    # Crear tabla de leyenda
    legend_table = Table(legend_data, hAlign='LEFT', spaceBefore=20, spaceAfter=20)
    return legend_table



def generate_pdf_with_front_page(pdf_path, model_name, data_name, metrics, class_names, class_colors, max_examples, metrics_classes, image_examples=None):
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Logo
    '''
    logo_path = '/app/assets/logo-cells-black.png'
    try:
        elements.append(Image(logo_path, width=150, height=50))
    except Exception:
        elements.append(Paragraph("Logo not found.", styles['Normal']))
    elements.append(Spacer(1, 20))
    '''
    logo = Image('./assets/logo-cells-black.png', width=(139*0.7), height=(38*0.7)) 
    logo.hAlign = 'LEFT' 
    elements.append(logo)

    # Portada
    elements.append(Paragraph(f"Inference Results", styles['Title']))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Model: <b>{model_name}</b>", styles['Normal']))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(f"Task: <b>Object Detection</b>", styles['Normal']))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(f"Dataset: <b>{data_name}</b>", styles['Normal']))
    elements.append(Spacer(1, 20))
   

    # Crear tabla de métricas
    metrics_data = [["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Class"]]
    metrics_data.append([
        f"{metrics['precision']:.3f}",
        f"{metrics['recall']:.3f}",
        f"{metrics['map@0.5']:.3f}",
        f"{metrics['map@0.5:0.95']:.3f}",
        "Global"
    ])

    # Añadir métricas por clase desde class_metrics
    for cls_name, cls_metrics in metrics_classes.items():
        metrics_data.append([
            f"{cls_metrics['precision']:.3f}",
            f"{cls_metrics['recall']:.3f}",
            f"{cls_metrics['map@0.5']:.3f}",
            f"{cls_metrics['map@0.5:0.95']:.3f}",
            cls_name
        ])

    # Configurar y añadir tabla de métricas
    metrics_table = Table(metrics_data, colWidths=[80, 80, 80, 80, 80])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#1b2a41")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (0, -1), HexColor("#eae9fa")),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))
    elements.append(create_signature_box())
    elements.append(Spacer(1, 40))


    # Filtrar e imprimir solo imágenes donde expected_classes y detected_classes no coincidan
    if image_examples:
        filtered_images = [
            (img_path, expected_classes, detected_classes)
            for img_path, expected_classes, detected_classes in image_examples
            if expected_classes != detected_classes
        ]

        for idx, (img_path, expected_classes, detected_classes) in enumerate(filtered_images[:max_examples]):
            elements.append(PageBreak())

            image_name = Path(img_path).name
            elements.append(Paragraph(f"{image_name}", styles['Title']))
            elements.append(Spacer(1, 10))

            # Expected Classes
            elements.append(Paragraph("Expected Classes:", styles['Normal']))
            if expected_classes:
                for cls, count in expected_classes.items():
                    elements.append(Paragraph(f"• Class {cls}: {count} occurrences", styles['Normal']))
            else:
                elements.append(Paragraph("<font color='red'>No expected classes</font>", styles['Normal']))
            elements.append(Spacer(1, 10))
            
            # Detected Classes
            elements.append(Paragraph("Detected Classes:", styles['Normal']))
            if detected_classes:
                for cls, count in detected_classes.items():
                    elements.append(Paragraph(f"• Class {cls}: {count} detections", styles['Normal']))
            else:
                elements.append(Paragraph("<font color='red'>Nothing detected</font>", styles['Normal']))
            elements.append(Spacer(1, 20))

             # Leyenda
            elements.append(Paragraph("Class Legend", styles['Normal']))
            elements.append(create_legend(class_colors))
            
            # Imagen
            try:
                elements.append(Image(img_path, width=400, height=300))
            except Exception as e:
                elements.append(Paragraph(f"Error loading image: {e}", styles['Normal']))
            elements.append(Spacer(1, 20))

    # Generar PDF
    doc.build(elements)
    print(f"PDF saved at: {pdf_path}")
