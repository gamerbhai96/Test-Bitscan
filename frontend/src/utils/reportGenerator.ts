import type { AnalysisResponse } from '../types/api';
import { utils, writeFile } from 'xlsx';
import jsPDF from 'jspdf';
import 'jspdf-autotable';

/**
 * Professional color palette for reports
 */
const REPORT_COLORS = {
  primary: [37, 99, 235],      // Blue
  secondary: [79, 70, 229],    // Indigo
  success: [16, 185, 129],     // Emerald
  warning: [245, 158, 11],     // Amber
  danger: [239, 68, 68],       // Red
  muted: [71, 85, 105],        // Slate
  light: [241, 245, 249],      // Light gray background
  white: [255, 255, 255],
  dark: [15, 23, 42]           // Dark slate
} as const;

/**
 * Layout constants
 */
const LAYOUT = {
  margin: 20,
  pageWidth: 210,
  pageHeight: 297,
  headerHeight: 35,
  footerHeight: 15
} as const;

/**
 * Initialize a professional PDF document
 */
const initProfessionalDocument = (): jsPDF => {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  doc.setFont('helvetica', 'normal');
  return doc;
};

/**
 * Draw professional header with company branding
 */
const drawProfessionalHeader = (doc: jsPDF, title: string, subtitle: string): number => {
  const { margin, pageWidth } = LAYOUT;

  // Header background
  doc.setFillColor(...REPORT_COLORS.primary);
  doc.rect(0, 0, pageWidth, LAYOUT.headerHeight, 'F');

  // Company logo area (placeholder for logo)
  doc.setFillColor(...REPORT_COLORS.white);
  doc.setDrawColor(...REPORT_COLORS.white);
  doc.roundedRect(margin, 8, 15, 15, 2, 2, 'F');

  // Logo text placeholder
  doc.setTextColor(...REPORT_COLORS.primary);
  doc.setFontSize(12);
  doc.setFont('helvetica', 'bold');
  doc.text('BS', margin + 7.5, 17, { align: 'center' });

  // Title
  doc.setTextColor(...REPORT_COLORS.white);
  doc.setFontSize(18);
  doc.setFont('helvetica', 'bold');
  doc.text(title, margin + 25, 15);

  // Subtitle
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text(subtitle, margin + 25, 23);

  // Date and time
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });
  const timeStr = now.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit'
  });

  doc.setFontSize(8);
  doc.setTextColor(240, 240, 240);
  doc.text(`${dateStr} at ${timeStr}`, pageWidth - margin, 12, { align: 'right' });

  return LAYOUT.headerHeight + 5;
};

/**
 * Draw professional footer with page numbers
 */
const drawProfessionalFooter = (doc: jsPDF, pageNum: number, totalPages: number): void => {
  const { margin, pageWidth, pageHeight, footerHeight } = LAYOUT;

  // Footer background
  doc.setFillColor(...REPORT_COLORS.light);
  doc.rect(0, pageHeight - footerHeight, pageWidth, footerHeight, 'F');

  // Footer line
  doc.setDrawColor(...REPORT_COLORS.primary);
  doc.setLineWidth(0.3);
  doc.line(margin, pageHeight - footerHeight, pageWidth - margin, pageHeight - footerHeight);

  // Page number
  doc.setTextColor(...REPORT_COLORS.muted);
  doc.setFontSize(8);
  doc.setFont('helvetica', 'normal');
  doc.text(`Page ${pageNum} of ${totalPages}`, pageWidth - margin, pageHeight - 5, { align: 'right' });

  // Disclaimer
  doc.setFontSize(7);
  doc.setTextColor(...REPORT_COLORS.muted);
  doc.text('CONFIDENTIAL - For authorized personnel only', margin, pageHeight - 5);
};

/**
 * Draw a professional metric card
 */
const drawMetricCard = (
  doc: jsPDF,
  x: number,
  y: number,
  width: number,
  height: number,
  label: string,
  value: string,
  color: readonly [number, number, number]
): void => {
  // Card background
  doc.setFillColor(...REPORT_COLORS.white);
  doc.setDrawColor(230, 235, 244);
  doc.setLineWidth(0.5);
  doc.roundedRect(x, y, width, height, 3, 3, 'FD');

  // Color accent bar
  doc.setFillColor(...color);
  doc.roundedRect(x, y, 4, height, 2, 2, 'F');

  // Label
  doc.setTextColor(...REPORT_COLORS.muted);
  doc.setFontSize(8);
  doc.setFont('helvetica', 'bold');
  doc.text(label.toUpperCase(), x + 8, y + 8);

  // Value
  doc.setTextColor(...REPORT_COLORS.dark);
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text(value, x + 8, y + height - 8);
};

/**
 * Draw a professional section header
 */
const drawSectionHeader = (doc: jsPDF, title: string, y: number): number => {
  const { margin, pageWidth } = LAYOUT;

  // Section title
  doc.setTextColor(...REPORT_COLORS.primary);
  doc.setFontSize(14);
  doc.setFont('helvetica', 'bold');
  doc.text(title, margin, y + 2);

  // Underline
  doc.setDrawColor(...REPORT_COLORS.primary);
  doc.setLineWidth(0.8);
  doc.line(margin, y + 6, Math.min(margin + 40, pageWidth - margin), y + 6);

  return y + 12;
};

/**
 * Format numbers professionally
 */
const formatNumber = (value: number, options?: Intl.NumberFormatOptions): string => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    ...options
  }).format(value);
};

const formatPercent = (value: number): string => {
  return `${(value * 100).toFixed(1)}%`;
};

/**
 * Generate and download report in Excel format
 */
export const downloadExcelReport = (analysis: AnalysisResponse, address: string) => {
  // Create workbook and worksheet
  const wb = utils.book_new();

  // Summary data
  const summaryData = [
    ['BitScan Analysis Report'],
    [''],
    ['Address', analysis.address],
    ['Risk Score', `${(analysis.risk_score * 100).toFixed(2)}%`],
    ['Risk Level', analysis.risk_level],
    ['Confidence', `${(analysis.confidence * 100).toFixed(2)}%`],
    ['Is Flagged', analysis.is_flagged ? 'Yes' : 'No'],
    ['Transaction Count', analysis.analysis_summary.transaction_count],
    ['Total Received (BTC)', analysis.analysis_summary.total_received_btc],
    ['Total Sent (BTC)', analysis.analysis_summary.total_sent_btc],
    ['Current Balance (BTC)', analysis.analysis_summary.current_balance_btc],
    [''],
    ['Generated on', new Date().toLocaleString()],
  ];

  // Create summary worksheet
  const wsSummary = utils.aoa_to_sheet(summaryData);
  utils.book_append_sheet(wb, wsSummary, 'Summary');

  // Risk factors data
  if (analysis.risk_factors && analysis.risk_factors.length > 0) {
    const riskFactorsData = [
      ['Risk Factors'],
      ...analysis.risk_factors.map((factor, index) => [`${index + 1}. ${factor}`])
    ];
    const wsRiskFactors = utils.aoa_to_sheet(riskFactorsData);
    utils.book_append_sheet(wb, wsRiskFactors, 'Risk Factors');
  }

  // Positive indicators data
  if (analysis.positive_indicators && analysis.positive_indicators.length > 0) {
    const positiveIndicatorsData = [
      ['Positive Indicators'],
      ...analysis.positive_indicators.map((indicator, index) => [`${index + 1}. ${indicator}`])
    ];
    const wsPositiveIndicators = utils.aoa_to_sheet(positiveIndicatorsData);
    utils.book_append_sheet(wb, wsPositiveIndicators, 'Positive Indicators');
  }

  // Data limitations data
  if (analysis.data_limitations) {
    const limitationsData = [
      ['Data Limitations'],
      ['Rate Limit Detected', analysis.data_limitations.rate_limit_detected ? 'Yes' : 'No'],
      ['Real Time Data', analysis.data_limitations.real_time_data ? 'Yes' : 'No'],
      ['API Status', analysis.data_limitations.api_status],
      ['Note', analysis.data_limitations.note || ''],
      ['Description', analysis.data_limitations.description || ''],
      ['Accuracy Note', analysis.data_limitations.accuracy_note || ''],
      ['Recommendation', analysis.data_limitations.recommendation || ''],
    ];
    const wsLimitations = utils.aoa_to_sheet(limitationsData);
    utils.book_append_sheet(wb, wsLimitations, 'Data Limitations');
  }

  // Download the file
  writeFile(wb, `BitScan_Report_${address.substring(0, 8)}_${new Date().toISOString().slice(0, 10)}.xlsx`);
};

/**
 * Generate and download clean executive-level PDF report
 */
export const downloadPdfReport = (analysis: AnalysisResponse, address: string) => {
  const doc = initProfessionalDocument();
  let currentY = drawProfessionalHeader(doc, 'BitScan Analysis Report', 'Cryptocurrency Risk Intelligence Report');

  // Address information section
  doc.setFillColor(...REPORT_COLORS.light);
  doc.roundedRect(LAYOUT.margin, currentY, LAYOUT.pageWidth - LAYOUT.margin * 2, 12, 2, 2, 'F');
  doc.setTextColor(...REPORT_COLORS.dark);
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  doc.text('WALLET ADDRESS:', LAYOUT.margin + 4, currentY + 8);
  doc.setFont('helvetica', 'normal');
  doc.text(address, LAYOUT.margin + 35, currentY + 8);
  currentY += 18;

  // Executive Summary Cards
  const cardWidth = (LAYOUT.pageWidth - LAYOUT.margin * 2 - 12) / 2; // Two cards per row
  const cardHeight = 28;

  // Risk Score Card
  const riskColor = analysis.risk_score > 0.7 ? REPORT_COLORS.danger :
                   analysis.risk_score > 0.4 ? REPORT_COLORS.warning : REPORT_COLORS.success;
  drawMetricCard(doc, LAYOUT.margin, currentY, cardWidth, cardHeight,
                'RISK SCORE', formatPercent(analysis.risk_score), riskColor);

  // Risk Level Card
  const levelColor = analysis.risk_level.toLowerCase().includes('high') ? REPORT_COLORS.danger :
                    analysis.risk_level.toLowerCase().includes('medium') ? REPORT_COLORS.warning : REPORT_COLORS.success;
  drawMetricCard(doc, LAYOUT.margin + cardWidth + 12, currentY, cardWidth, cardHeight,
                'RISK LEVEL', analysis.risk_level.toUpperCase(), levelColor);

  currentY += cardHeight + 8;

  // Additional Metrics Cards
  drawMetricCard(doc, LAYOUT.margin, currentY, cardWidth, cardHeight,
                'CONFIDENCE', formatPercent(analysis.confidence), REPORT_COLORS.secondary);

  drawMetricCard(doc, LAYOUT.margin + cardWidth + 12, currentY, cardWidth, cardHeight,
                'TRANSACTIONS', analysis.analysis_summary.transaction_count.toLocaleString(), REPORT_COLORS.primary);

  currentY += cardHeight + 8;

  // Balance and Status Cards
  drawMetricCard(doc, LAYOUT.margin, currentY, cardWidth, cardHeight,
                'CURRENT BALANCE',
                `${formatNumber(analysis.analysis_summary.current_balance_btc, {
                  minimumFractionDigits: 4,
                  maximumFractionDigits: 8
                })} BTC`, REPORT_COLORS.secondary);

  drawMetricCard(doc, LAYOUT.margin + cardWidth + 12, currentY, cardWidth, cardHeight,
                'STATUS', analysis.is_flagged ? 'FLAGGED' : 'CLEAR',
                analysis.is_flagged ? REPORT_COLORS.danger : REPORT_COLORS.success);

  currentY += cardHeight + 15;

  // Risk Intelligence Section
  if (analysis.risk_factors && analysis.risk_factors.length > 0) {
    currentY = drawSectionHeader(doc, 'Risk Intelligence Assessment', currentY);

    // Add descriptive text
    doc.setFontSize(10);
    doc.setTextColor(...REPORT_COLORS.muted);
    doc.setFont('helvetica', 'normal');
    doc.text('The following risk factors were identified during the comprehensive blockchain analysis:', LAYOUT.margin, currentY + 6);
    currentY += 12;

    const riskTableData = analysis.risk_factors.map((factor, index) => [
      `${index + 1}`,
      factor
    ]);

    (doc as any).autoTable({
      startY: currentY,
      head: [['#', 'Identified Risk Factor']],
      body: riskTableData,
      margin: { left: LAYOUT.margin, right: LAYOUT.margin },
      theme: 'plain',
      styles: {
        fontSize: 10,
        cellPadding: { top: 6, bottom: 6, left: 5, right: 5 },
        textColor: REPORT_COLORS.dark,
        lineWidth: 0.1,
        lineColor: [226, 232, 240]
      },
      headStyles: {
        fillColor: REPORT_COLORS.primary,
        textColor: REPORT_COLORS.white,
        fontStyle: 'bold',
        halign: 'left',
        fontSize: 10
      },
      alternateRowStyles: {
        fillColor: REPORT_COLORS.light
      }
    });

    currentY = ((doc as any).lastAutoTable?.finalY ?? currentY) + 15;
  } else {
    currentY = drawSectionHeader(doc, 'Risk Intelligence Assessment', currentY);
    doc.setFontSize(10);
    doc.setTextColor(...REPORT_COLORS.success);
    doc.setFont('helvetica', 'normal');
    doc.text('✓ No significant risk factors were identified during the comprehensive analysis.', LAYOUT.margin, currentY + 6);
    currentY += 20;
  }

  // Positive Indicators Section
  if (analysis.positive_indicators && analysis.positive_indicators.length > 0) {
    currentY = drawSectionHeader(doc, 'Mitigating Factors', currentY);

    // Add descriptive text
    doc.setFontSize(10);
    doc.setTextColor(...REPORT_COLORS.muted);
    doc.setFont('helvetica', 'normal');
    doc.text('The following positive indicators were identified that may reduce overall risk:', LAYOUT.margin, currentY + 6);
    currentY += 12;

    const positiveTableData = analysis.positive_indicators.map((indicator, index) => [
      `${index + 1}`,
      indicator
    ]);

    (doc as any).autoTable({
      startY: currentY,
      head: [['#', 'Positive Indicator']],
      body: positiveTableData,
      margin: { left: LAYOUT.margin, right: LAYOUT.margin },
      theme: 'plain',
      styles: {
        fontSize: 10,
        cellPadding: { top: 6, bottom: 6, left: 5, right: 5 },
        textColor: REPORT_COLORS.dark,
        lineWidth: 0.1,
        lineColor: [226, 232, 240]
      },
      headStyles: {
        fillColor: REPORT_COLORS.success,
        textColor: REPORT_COLORS.white,
        fontStyle: 'bold',
        halign: 'left',
        fontSize: 10
      },
      alternateRowStyles: {
        fillColor: REPORT_COLORS.light
      }
    });

    currentY = ((doc as any).lastAutoTable?.finalY ?? currentY) + 15;
  }

  // Data Quality & Limitations Section
  if (analysis.data_limitations) {
    currentY = drawSectionHeader(doc, 'Data Quality Assessment', currentY);

    const limitationsTableData = [
      ['Rate Limiting', analysis.data_limitations.rate_limit_detected ? 'Detected' : 'Not Detected'],
      ['Real-time Data', analysis.data_limitations.real_time_data ? 'Available' : 'Not Available'],
      ['API Status', analysis.data_limitations.api_status],
      ['Accuracy Note', analysis.data_limitations.accuracy_note || 'N/A'],
      ['Recommendation', analysis.data_limitations.recommendation || 'N/A']
    ];

    (doc as any).autoTable({
      startY: currentY,
      head: [['Assessment', 'Status/Details']],
      body: limitationsTableData,
      margin: { left: LAYOUT.margin, right: LAYOUT.margin },
      theme: 'striped',
      styles: {
        fontSize: 10,
        cellPadding: { top: 5, bottom: 5, left: 5, right: 5 },
        textColor: REPORT_COLORS.dark
      },
      headStyles: {
        fillColor: REPORT_COLORS.warning,
        textColor: REPORT_COLORS.white,
        fontStyle: 'bold',
        halign: 'left'
      },
      alternateRowStyles: { fillColor: REPORT_COLORS.light },
      tableLineColor: [226, 232, 240],
      tableLineWidth: 0.3
    });

    currentY = ((doc as any).lastAutoTable?.finalY ?? currentY) + 10;
  }

  // Add footer to all pages
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    drawProfessionalFooter(doc, i, totalPages);
  }

  // Save the professional report
  doc.save(`BitScan_Professional_Report_${address.substring(0, 8)}_${new Date().toISOString().slice(0, 10)}.pdf`);
};

/**
 * Generate and download clean executive-level PDF report (alternative layout)
 */
export const downloadPdfReportWithCharts = async (analysis: AnalysisResponse, address: string) => {
  const doc = initProfessionalDocument();
  let currentY = drawProfessionalHeader(doc, 'BitScan Analysis Report', 'Cryptocurrency Risk Intelligence Report');

  // Address information section
  doc.setFillColor(...REPORT_COLORS.light);
  doc.roundedRect(LAYOUT.margin, currentY, LAYOUT.pageWidth - LAYOUT.margin * 2, 12, 2, 2, 'F');
  doc.setTextColor(...REPORT_COLORS.dark);
  doc.setFontSize(11);
  doc.setFont('helvetica', 'bold');
  doc.text('SUBJECT WALLET ADDRESS:', LAYOUT.margin + 4, currentY + 8);
  doc.setFont('helvetica', 'normal');
  doc.text(address, LAYOUT.margin + 45, currentY + 8);
  currentY += 18;

  // Executive Summary Section
  currentY = drawSectionHeader(doc, 'Executive Summary', currentY);

  // Executive Summary Cards - Professional Layout
  const cardWidth = (LAYOUT.pageWidth - LAYOUT.margin * 2 - 16) / 2; // Two cards per row
  const cardHeight = 30;

  // Primary Risk Metrics
  const riskColor = analysis.risk_score > 0.7 ? REPORT_COLORS.danger :
                   analysis.risk_score > 0.4 ? REPORT_COLORS.warning : REPORT_COLORS.success;
  drawMetricCard(doc, LAYOUT.margin, currentY, cardWidth, cardHeight,
                'OVERALL RISK SCORE', formatPercent(analysis.risk_score), riskColor);

  const levelColor = analysis.risk_level.toLowerCase().includes('high') ? REPORT_COLORS.danger :
                    analysis.risk_level.toLowerCase().includes('medium') ? REPORT_COLORS.warning : REPORT_COLORS.success;
  drawMetricCard(doc, LAYOUT.margin + cardWidth + 16, currentY, cardWidth, cardHeight,
                'RISK CLASSIFICATION', analysis.risk_level.toUpperCase(), levelColor);

  currentY += cardHeight + 10;

  // Additional Key Metrics
  drawMetricCard(doc, LAYOUT.margin, currentY, cardWidth, cardHeight,
                'ANALYSIS CONFIDENCE', formatPercent(analysis.confidence), REPORT_COLORS.secondary);

  drawMetricCard(doc, LAYOUT.margin + cardWidth + 16, currentY, cardWidth, cardHeight,
                'WALLET STATUS', analysis.is_flagged ? 'REQUIRES ATTENTION' : 'APPROVED',
                analysis.is_flagged ? REPORT_COLORS.danger : REPORT_COLORS.success);

  currentY += cardHeight + 10;

  // Financial Overview
  drawMetricCard(doc, LAYOUT.margin, currentY, cardWidth, cardHeight,
                'CURRENT BALANCE',
                `${formatNumber(analysis.analysis_summary.current_balance_btc, {
                  minimumFractionDigits: 4,
                  maximumFractionDigits: 8
                })} BTC`, REPORT_COLORS.primary);

  drawMetricCard(doc, LAYOUT.margin + cardWidth + 16, currentY, cardWidth, cardHeight,
                'TOTAL TRANSACTIONS', analysis.analysis_summary.transaction_count.toLocaleString(), REPORT_COLORS.primary);

  currentY += cardHeight + 20;

  // Risk Intelligence Section
  if (analysis.risk_factors && analysis.risk_factors.length > 0) {
    currentY = drawSectionHeader(doc, 'Risk Intelligence Assessment', currentY);

    // Add descriptive text
    doc.setFontSize(10);
    doc.setTextColor(...REPORT_COLORS.muted);
    doc.setFont('helvetica', 'normal');
    doc.text('The following risk factors were identified during the comprehensive blockchain analysis:', LAYOUT.margin, currentY + 6);
    currentY += 12;

    const riskTableData = analysis.risk_factors.map((factor, index) => [
      `${index + 1}`,
      factor
    ]);

    (doc as any).autoTable({
      startY: currentY,
      head: [['#', 'Identified Risk Factor']],
      body: riskTableData,
      margin: { left: LAYOUT.margin, right: LAYOUT.margin },
      theme: 'plain',
      styles: {
        fontSize: 10,
        cellPadding: { top: 6, bottom: 6, left: 5, right: 5 },
        textColor: REPORT_COLORS.dark,
        lineWidth: 0.1,
        lineColor: [226, 232, 240]
      },
      headStyles: {
        fillColor: REPORT_COLORS.primary,
        textColor: REPORT_COLORS.white,
        fontStyle: 'bold',
        halign: 'left',
        fontSize: 10
      },
      alternateRowStyles: {
        fillColor: REPORT_COLORS.light
      }
    });

    currentY = ((doc as any).lastAutoTable?.finalY ?? currentY) + 15;
  } else {
    currentY = drawSectionHeader(doc, 'Risk Intelligence Assessment', currentY);
    doc.setFontSize(10);
    doc.setTextColor(...REPORT_COLORS.success);
    doc.setFont('helvetica', 'normal');
    doc.text('✓ No significant risk factors were identified during the comprehensive analysis.', LAYOUT.margin, currentY + 6);
    currentY += 20;
  }

  // Mitigating Factors Section
  if (analysis.positive_indicators && analysis.positive_indicators.length > 0) {
    currentY = drawSectionHeader(doc, 'Mitigating Factors', currentY);

    // Add descriptive text
    doc.setFontSize(10);
    doc.setTextColor(...REPORT_COLORS.muted);
    doc.setFont('helvetica', 'normal');
    doc.text('The following positive indicators were identified that may reduce overall risk:', LAYOUT.margin, currentY + 6);
    currentY += 12;

    const positiveTableData = analysis.positive_indicators.map((indicator, index) => [
      `${index + 1}`,
      indicator
    ]);

    (doc as any).autoTable({
      startY: currentY,
      head: [['#', 'Positive Indicator']],
      body: positiveTableData,
      margin: { left: LAYOUT.margin, right: LAYOUT.margin },
      theme: 'plain',
      styles: {
        fontSize: 10,
        cellPadding: { top: 6, bottom: 6, left: 5, right: 5 },
        textColor: REPORT_COLORS.dark,
        lineWidth: 0.1,
        lineColor: [226, 232, 240]
      },
      headStyles: {
        fillColor: REPORT_COLORS.success,
        textColor: REPORT_COLORS.white,
        fontStyle: 'bold',
        halign: 'left',
        fontSize: 10
      },
      alternateRowStyles: {
        fillColor: REPORT_COLORS.light
      }
    });

    currentY = ((doc as any).lastAutoTable?.finalY ?? currentY) + 15;
  }

  // Data Quality & Methodology Section
  currentY = drawSectionHeader(doc, 'Data Quality & Methodology', currentY);

  // Add methodology text
  doc.setFontSize(10);
  doc.setTextColor(...REPORT_COLORS.dark);
  doc.setFont('helvetica', 'normal');
  const methodologyText = 'This analysis was conducted using advanced blockchain analytics and machine learning algorithms to assess cryptocurrency wallet behavior patterns, transaction history, and risk indicators.';
  const splitText = doc.splitTextToSize(methodologyText, LAYOUT.pageWidth - LAYOUT.margin * 2);
  doc.text(splitText, LAYOUT.margin, currentY + 6);
  currentY += splitText.length * 5 + 10;

  if (analysis.data_limitations) {
    // Data limitations table
    const limitationsTableData = [
      ['Data Source Status', analysis.data_limitations.api_status || 'Operational'],
      ['Real-time Data Access', analysis.data_limitations.real_time_data ? 'Available' : 'Limited'],
      ['Rate Limiting Detected', analysis.data_limitations.rate_limit_detected ? 'Yes' : 'No'],
      ['Accuracy Assessment', analysis.data_limitations.accuracy_note || 'High Confidence'],
      ['Analysis Coverage', 'Comprehensive blockchain analysis']
    ];

    (doc as any).autoTable({
      startY: currentY,
      head: [['Assessment Category', 'Status/Details']],
      body: limitationsTableData,
      margin: { left: LAYOUT.margin, right: LAYOUT.margin },
      theme: 'plain',
      styles: {
        fontSize: 9,
        cellPadding: { top: 4, bottom: 4, left: 5, right: 5 },
        textColor: REPORT_COLORS.dark,
        lineWidth: 0.1,
        lineColor: [226, 232, 240]
      },
      headStyles: {
        fillColor: REPORT_COLORS.warning,
        textColor: REPORT_COLORS.white,
        fontStyle: 'bold',
        halign: 'left',
        fontSize: 9
      },
      alternateRowStyles: {
        fillColor: REPORT_COLORS.light
      }
    });

    currentY = ((doc as any).lastAutoTable?.finalY ?? currentY) + 15;
  }

  // Professional Recommendations Section
  currentY = drawSectionHeader(doc, 'Recommendations', currentY);

  doc.setFontSize(10);
  doc.setTextColor(...REPORT_COLORS.dark);
  doc.setFont('helvetica', 'normal');

  const recommendations = [];
  if (analysis.risk_score > 0.7) {
    recommendations.push('• Immediate review and enhanced due diligence recommended');
    recommendations.push('• Consider transaction monitoring and additional verification');
  } else if (analysis.risk_score > 0.4) {
    recommendations.push('• Moderate risk identified - proceed with caution');
    recommendations.push('• Additional verification steps may be beneficial');
  } else {
    recommendations.push('• Low risk profile identified');
    recommendations.push('• Standard due diligence procedures recommended');
  }

  if (analysis.data_limitations?.rate_limit_detected) {
    recommendations.push('• Data completeness may be limited due to API rate limiting');
    recommendations.push('• Consider follow-up analysis for complete transaction history');
  }

  recommendations.forEach((rec, index) => {
    doc.text(rec, LAYOUT.margin, currentY + 6 + (index * 5));
  });

  currentY += recommendations.length * 5 + 20;

  // Add footer to all pages
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    drawProfessionalFooter(doc, i, totalPages);
  }

  // Save the clean professional report
  doc.save(`BitScan_Executive_Report_${address.substring(0, 8)}_${new Date().toISOString().slice(0, 10)}.pdf`);
};