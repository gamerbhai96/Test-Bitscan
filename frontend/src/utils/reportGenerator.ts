import type { AnalysisResponse } from '../types/api';
import { utils, writeFile } from 'xlsx';
import jsPDF from 'jspdf';
import 'jspdf-autotable';

/**
 * Professional color palette matching SecureDApp's blockchain security brand
 */
const REPORT_COLORS = {
  primary: [30, 58, 138],      // SecureDApp professional dark blue #1e3a8a
  secondary: [14, 165, 233],   // SecureDApp sky blue #0ea5e9
  success: [34, 197, 94],      // Professional green
  warning: [251, 146, 60],     // Professional amber
  danger: [239, 68, 68],       // Professional red
  muted: [71, 85, 105],        // Cool slate gray
  light: [248, 250, 252],      // Very light background
  white: [255, 255, 255],
  dark: [15, 23, 42],          // Professional dark slate
  accent: [139, 92, 246]       // Professional purple accent
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
 * Add SecureDApp logo to the PDF document using the local PNG file
 */
const addSecureDAppLogo = async (doc: jsPDF, x: number, y: number, width: number, height: number): Promise<void> => {
  try {
    // Use the local PNG logo (CORS-protected SVG not accessible)
    const response = await fetch('/logo.png');
    const blob = await response.blob();

    // Convert blob to base64
    const reader = new FileReader();
    const base64Promise = new Promise<string>((resolve, reject) => {
      reader.onload = () => {
        const result = reader.result as string;
        const base64 = result.split(',')[1]; // Remove data:image/png;base64, prefix
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

    const base64 = await base64Promise;
    const dataUrl = `data:image/png;base64,${base64}`;

    // Add the local PNG logo with proper dimensions (matching website size)
    doc.addImage(dataUrl, 'PNG', x, y, width, height);
  } catch (error) {
    console.warn('Failed to load SecureDApp logo:', error);

    // Fallback: Draw a professional text-based logo placeholder
    doc.setFillColor(30, 58, 138);
    doc.setDrawColor(30, 58, 138);
    doc.roundedRect(x, y, width, height, 4, 4, 'FD');

    // Add SecureDApp text
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.text('SECUREDAPP', x + width / 2, y + height / 2, { align: 'center' });
  }
};

/**
 * Draw professional header matching SecureDApp website layout (clean and simple)
 */
const drawProfessionalHeader = async (doc: jsPDF, title: string, subtitle: string): Promise<number> => {
  const { margin, pageWidth } = LAYOUT;

  // Clean professional header background
  doc.setFillColor(30, 58, 138); // Professional dark blue #1e3a8a
  doc.rect(0, 0, pageWidth, LAYOUT.headerHeight, 'F');

  // Simple accent line at bottom
  doc.setFillColor(14, 165, 233); // Sky blue accent #0ea5e9
  doc.rect(0, LAYOUT.headerHeight - 2, pageWidth, 2, 'F');

  // Logo stretched horizontally to utilize left space better
  const logoWidth = 55; // Wider horizontal stretch for better space utilization
  const logoHeight = 30; // Maintain reasonable height
  const logoX = margin;
  const logoY = (LAYOUT.headerHeight - logoHeight) / 2; // Perfectly centered vertically

  await addSecureDAppLogo(doc, logoX, logoY, logoWidth, logoHeight);

  // Adjust text positioning to accommodate wider logo
  const textStartX = logoX + logoWidth + 18;

  // Professional title typography
  doc.setTextColor(...REPORT_COLORS.white);
  doc.setFontSize(16);
  doc.setFont('helvetica', 'bold');
  doc.text(title, textStartX, LAYOUT.headerHeight / 2);

  // Clean subtitle
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.text(subtitle, textStartX, LAYOUT.headerHeight / 2 + 6);

  // Minimal date on right
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });

  doc.setFontSize(8);
  doc.setTextColor(220, 230, 255);
  doc.text(dateStr, pageWidth - margin, LAYOUT.headerHeight / 2, { align: 'right' });

  return LAYOUT.headerHeight + 5;
};

/**
 * Draw professional footer with SecureDApp branding
 */
const drawProfessionalFooter = (doc: jsPDF, pageNum: number, totalPages: number): void => {
  const { margin, pageWidth, pageHeight, footerHeight } = LAYOUT;

  // Footer background with SecureDApp blue
  doc.setFillColor(0, 123, 255);
  doc.rect(0, pageHeight - footerHeight, pageWidth, footerHeight, 'F');

  // Footer line
  doc.setDrawColor(0, 188, 212);
  doc.setLineWidth(0.3);
  doc.line(margin, pageHeight - footerHeight, pageWidth - margin, pageHeight - footerHeight);

  // Page number with SecureDApp styling
  doc.setTextColor(240, 248, 255);
  doc.setFontSize(8);
  doc.setFont('helvetica', 'normal');
  doc.text(`Page ${pageNum} of ${totalPages}`, pageWidth - margin, pageHeight - 5, { align: 'right' });

  // SecureDApp branding
  doc.setFontSize(7);
  doc.setTextColor(200, 220, 255);
  doc.text('Powered by SecureDApp.io', margin, pageHeight - 5);

  // Professional disclaimer
  doc.setFontSize(6);
  doc.setTextColor(180, 200, 255);
  doc.text('Blockchain Security Intelligence Report - For Professional Use Only', pageWidth / 2, pageHeight - 5, { align: 'center' });
};

/**
 * Draw a professional metric card with enhanced styling
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
  // Card background with subtle shadow effect
  doc.setFillColor(250, 250, 250);
  doc.setDrawColor(230, 235, 244);
  doc.setLineWidth(0.3);
  doc.roundedRect(x + 0.5, y + 0.5, width, height, 4, 4, 'FD');

  // Main card background
  doc.setFillColor(...REPORT_COLORS.white);
  doc.setDrawColor(226, 232, 240);
  doc.setLineWidth(0.8);
  doc.roundedRect(x, y, width, height, 4, 4, 'FD');

  // Enhanced color accent bar with gradient effect
  doc.setFillColor(...color);
  doc.roundedRect(x, y, 6, height, 3, 3, 'F');

  // Secondary accent for depth
  doc.setFillColor(Math.max(0, color[0] - 20), Math.max(0, color[1] - 20), Math.max(0, color[2] - 20));
  doc.roundedRect(x, y + height - 3, 6, 3, 0, 0, 'F');

  // Label with improved typography
  doc.setTextColor(...REPORT_COLORS.muted);
  doc.setFontSize(7);
  doc.setFont('helvetica', 'bold');
  doc.text(label.toUpperCase(), x + 10, y + 8);

  // Value with enhanced styling
  doc.setTextColor(...REPORT_COLORS.dark);
  doc.setFontSize(13);
  doc.setFont('helvetica', 'bold');
  doc.text(value, x + 10, y + height - 6);

  // Subtle border accent
  doc.setDrawColor(...color);
  doc.setLineWidth(0.3);
  doc.roundedRect(x + 0.5, y + 0.5, width - 1, height - 1, 3.5, 3.5, 'S');
};

/**
 * Draw a professional section header with enhanced styling (prevents duplicates)
 */
const drawSectionHeader = (doc: jsPDF, title: string, y: number): number => {
  const { margin, pageWidth } = LAYOUT;

  // Check if we're at the very top of a page (likely a page break issue)
  const currentPage = doc.getCurrentPageInfo().pageNumber;
  const pageHeight = LAYOUT.pageHeight;
  const headerHeight = LAYOUT.headerHeight + LAYOUT.footerHeight;

  // If we're too close to the top of the page, add some spacing to prevent header duplication
  let adjustedY = y;
  if (y < headerHeight + 20) { // If we're too close to the header area
    adjustedY = headerHeight + 20; // Move down
  }

  // Section title with enhanced styling
  doc.setTextColor(...REPORT_COLORS.primary);
  doc.setFontSize(16);
  doc.setFont('helvetica', 'bold');
  doc.text(title, margin, adjustedY + 3);

  // Professional underline with SecureDApp gradient effect
  doc.setDrawColor(0, 123, 255);
  doc.setLineWidth(1.2);
  doc.line(margin, adjustedY + 8, Math.min(margin + 50, pageWidth - margin), adjustedY + 8);

  // Secondary accent line
  doc.setDrawColor(0, 188, 212);
  doc.setLineWidth(0.6);
  doc.line(margin, adjustedY + 10, Math.min(margin + 40, pageWidth - margin), adjustedY + 10);

  return adjustedY + 15;
};

/**
 * Format Bitcoin address for display with truncation
 */
const formatBitcoinAddress = (address: string, maxLength: number = 40): string => {
  if (address.length <= maxLength) return address;
  const prefixLength = Math.floor((maxLength - 3) / 2);
  const suffixLength = maxLength - 3 - prefixLength;
  return `${address.substring(0, prefixLength)}...${address.substring(address.length - suffixLength)}`;
};

/**
 * Draw wallet address section with proper formatting and professional spacing
 */
const drawWalletAddressSection = (doc: jsPDF, address: string, label: string, currentY: number): number => {
  const { margin, pageWidth } = LAYOUT;

  // Professional background box with subtle styling
  doc.setFillColor(248, 250, 252); // Very light gray
  doc.setDrawColor(226, 232, 240); // Light border
  doc.setLineWidth(0.5);
  doc.roundedRect(margin, currentY, pageWidth - margin * 2, 18, 4, 4, 'FD');

  // Label with professional typography
  doc.setTextColor(...REPORT_COLORS.primary);
  doc.setFontSize(12);
  doc.setFont('helvetica', 'bold');
  doc.text(label, margin + 6, currentY + 12);

  // Address with better spacing and formatting
  doc.setTextColor(...REPORT_COLORS.dark);
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');

  const maxAddressWidth = pageWidth - margin * 2 - 90; // More space for label (90px)
  const formattedAddress = formatBitcoinAddress(address, 55); // Allow up to 55 chars before truncation

  // Check if address fits, if not, split into two lines with better spacing
  const addressWidth = doc.getTextWidth(formattedAddress);
  if (addressWidth <= maxAddressWidth) {
    doc.text(formattedAddress, margin + 90, currentY + 12);
  } else {
    // Split long address into two lines with better line spacing
    const midPoint = Math.floor(formattedAddress.length / 2);
    const firstLine = formattedAddress.substring(0, midPoint);
    const secondLine = formattedAddress.substring(midPoint);

    doc.text(firstLine, margin + 90, currentY + 9);
    doc.text(secondLine, margin + 90, currentY + 15);

    // Increase height for two-line address with better spacing
    return currentY + 22;
  }

  return currentY + 20;
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
    ['Subject Wallet Address', formatBitcoinAddress(address, 60)],
    ['Risk Score', `${(analysis.risk_score * 100).toFixed(2)}%`],
    ['Risk Level', analysis.risk_level],
    ['Confidence', `${(analysis.confidence * 100).toFixed(2)}%`],
    ['Status', analysis.is_flagged ? 'FLAGGED' : 'CLEAR'],
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
  writeFile(wb, `SecureDApp_Security_Report_${address.substring(0, 8)}_${new Date().toISOString().slice(0, 10)}.xlsx`);
};

/**
 * Generate and download clean executive-level PDF report
 */
export const downloadPdfReport = async (analysis: AnalysisResponse, address: string) => {
  const doc = initProfessionalDocument();
  let currentY = await drawProfessionalHeader(doc, 'Blockchain Security Analysis', 'Comprehensive Risk Intelligence Report');

  // Address information section
  currentY = drawWalletAddressSection(doc, address, 'WALLET ADDRESS:', currentY);

  // Add professional spacing before metrics
  currentY += 8;

  // Executive Summary Cards
  const cardWidth = (LAYOUT.pageWidth - LAYOUT.margin * 2 - 12) / 2; // Two cards per row
  const cardHeight = 32; // Increased height for better visual balance

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

    // Add descriptive text with better spacing
    doc.setFontSize(10);
    doc.setTextColor(...REPORT_COLORS.muted);
    doc.setFont('helvetica', 'normal');
    doc.text('The following risk factors were identified during the comprehensive blockchain analysis:', LAYOUT.margin, currentY + 6);
    currentY += 14;

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
  doc.save(`SecureDApp_Security_Analysis_${address.substring(0, 8)}_${new Date().toISOString().slice(0, 10)}.pdf`);
};

/**
 * Generate and download clean executive-level PDF report (alternative layout)
 */
export const downloadPdfReportWithCharts = async (analysis: AnalysisResponse, address: string) => {
  const doc = initProfessionalDocument();
  let currentY = await drawProfessionalHeader(doc, 'Blockchain Security Intelligence', 'Advanced Risk Assessment Report');

  // Address information section
  currentY = drawWalletAddressSection(doc, address, 'SUBJECT WALLET ADDRESS:', currentY);

  // Add professional spacing before summary section
  currentY += 10;

  // Executive Summary Section
  currentY = drawSectionHeader(doc, 'Executive Summary', currentY);

  // Executive Summary Cards - Professional Layout
  const cardWidth = (LAYOUT.pageWidth - LAYOUT.margin * 2 - 16) / 2; // Two cards per row
  const cardHeight = 35; // Enhanced height for professional appearance

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

  // Risk Intelligence Section - Always include this section
  currentY = drawSectionHeader(doc, 'Risk Intelligence Assessment', currentY);

  if (analysis.risk_factors && analysis.risk_factors.length > 0) {
    // Add descriptive text for risk factors
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
    // No risk factors found - show positive message
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
  doc.save(`SecureDApp_Intelligence_Report_${address.substring(0, 8)}_${new Date().toISOString().slice(0, 10)}.pdf`);
};