import type { AnalysisResponse } from '../types/api';
import type { WalletTimeSeriesResponse } from '../types/timeseries';
import { utils, writeFile } from 'xlsx';
import jsPDF from 'jspdf';
import 'jspdf-autotable';
import { BitScanAPI } from '../services/api';

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
 * Generate and download report in PDF format
 */
export const downloadPdfReport = (analysis: AnalysisResponse, address: string) => {
  const doc = new jsPDF();

  // Title
  doc.setFontSize(20);
  doc.text('BitScan Analysis Report', 105, 20, { align: 'center' });

  // Subtitle
  doc.setFontSize(12);
  doc.setTextColor(100);
  doc.text(`Address: ${address}`, 105, 30, { align: 'center' });
  doc.text(`Generated on: ${new Date().toLocaleString()}`, 105, 37, { align: 'center' });

  // Summary section
  doc.setFontSize(16);
  doc.setTextColor(0);
  doc.text('Summary', 20, 50);

  doc.setFontSize(12);
  doc.text(`Risk Score: ${(analysis.risk_score * 100).toFixed(2)}%`, 20, 60);
  doc.text(`Risk Level: ${analysis.risk_level}`, 20, 67);
  doc.text(`Confidence: ${(analysis.confidence * 100).toFixed(2)}%`, 20, 74);
  doc.text(`Is Flagged: ${analysis.is_flagged ? 'Yes' : 'No'}`, 20, 81);
  doc.text(`Transaction Count: ${analysis.analysis_summary.transaction_count}`, 20, 88);
  doc.text(`Current Balance: ${analysis.analysis_summary.current_balance_btc.toFixed(8)} BTC`, 20, 95);
  // Include unique counterparties if available from time-series report
  // Note: This function does not fetch time-series; value may be added by callers if needed.

  // Risk factors section
  if (analysis.risk_factors && analysis.risk_factors.length > 0) {
    doc.setFontSize(16);
    doc.text('Risk Factors', 20, 110);

    doc.setFontSize(12);
    const riskFactors = analysis.risk_factors.map((factor, index) => [
      `${index + 1}.`,
      factor
    ]);

    (doc as any).autoTable({
      startY: 115,
      head: [['#', 'Risk Factor']],
      body: riskFactors,
      theme: 'grid',
      styles: { fontSize: 10 },
      headStyles: { fillColor: [37, 99, 235] },
      margin: { horizontal: 20 }
    });
  }

  // Positive indicators section
  const finalY = (doc as any).lastAutoTable?.finalY || 130;

  if (analysis.positive_indicators && analysis.positive_indicators.length > 0) {
    doc.setFontSize(16);
    doc.text('Positive Indicators', 20, finalY + 15);

    doc.setFontSize(12);
    const positiveIndicators = analysis.positive_indicators.map((indicator, index) => [
      `${index + 1}.`,
      indicator
    ]);

    (doc as any).autoTable({
      startY: finalY + 20,
      head: [['#', 'Positive Indicator']],
      body: positiveIndicators,
      theme: 'grid',
      styles: { fontSize: 10 },
      headStyles: { fillColor: [5, 150, 105] },
      margin: { horizontal: 20 }
    });
  }

  // Data limitations section
  const finalY2 = (doc as any).lastAutoTable?.finalY || finalY + 35;

  if (analysis.data_limitations) {
    doc.setFontSize(16);
    doc.text('Data Limitations', 20, finalY2 + 15);

    doc.setFontSize(12);
    const limitations = [
      ['Rate Limit Detected', analysis.data_limitations.rate_limit_detected ? 'Yes' : 'No'],
      ['Real Time Data', analysis.data_limitations.real_time_data ? 'Yes' : 'No'],
      ['API Status', analysis.data_limitations.api_status],
      ['Note', analysis.data_limitations.note || 'N/A'],
      ['Description', analysis.data_limitations.description || 'N/A'],
      ['Accuracy Note', analysis.data_limitations.accuracy_note || 'N/A'],
      ['Recommendation', analysis.data_limitations.recommendation || 'N/A']
    ];

    (doc as any).autoTable({
      startY: finalY2 + 20,
      head: [['Limitation', 'Value']],
      body: limitations,
      theme: 'grid',
      styles: { fontSize: 10 },
      headStyles: { fillColor: [245, 158, 11] },
      margin: { horizontal: 20 }
    });
  }

  // Download the file
  doc.save(`BitScan_Report_${address.substring(0, 8)}_${new Date().toISOString().slice(0, 10)}.pdf`);
};

/**
 * Generate and download report in PDF format with embedded charts (weekly and monthly)
 */
export const downloadPdfReportWithCharts = async (analysis: AnalysisResponse, address: string) => {
  const doc = new jsPDF();

  // Title
  doc.setFontSize(20);
  doc.text('BitScan Analysis Report', 105, 20, { align: 'center' });

  // Subtitle
  doc.setFontSize(12);
  doc.setTextColor(100);
  doc.text(`Address: ${address}`, 105, 30, { align: 'center' });
  doc.text(`Generated on: ${new Date().toLocaleString()}`, 105, 37, { align: 'center' });

  // Summary (reusing the original generator contents briefly)
  doc.setFontSize(16);
  doc.setTextColor(0);
  doc.text('Summary', 20, 50);

  doc.setFontSize(12);
  doc.text(`Risk Score: ${(analysis.risk_score * 100).toFixed(2)}%`, 20, 60);
  doc.text(`Risk Level: ${analysis.risk_level}`, 20, 67);
  doc.text(`Confidence: ${(analysis.confidence * 100).toFixed(2)}%`, 20, 74);
  doc.text(`Is Flagged: ${analysis.is_flagged ? 'Yes' : 'No'}`, 20, 81);
  doc.text(`Transaction Count: ${analysis.analysis_summary.transaction_count}`, 20, 88);
  doc.text(`Current Balance: ${analysis.analysis_summary.current_balance_btc.toFixed(8)} BTC`, 20, 95);

  // Fetch time series for charts
  let weekly: WalletTimeSeriesResponse | null = null;
  let monthly: WalletTimeSeriesResponse | null = null;
  try {
    weekly = await BitScanAPI.getWalletTimeSeries(address, 365, 'week');
  } catch {}
  try {
    monthly = await BitScanAPI.getWalletTimeSeries(address, 365 * 3, 'month');
  } catch {}

  // Show unique counterparties if available
  const uniq = (weekly?.summary as any)?.unique_counterparties ?? (monthly?.summary as any)?.unique_counterparties;
  if (typeof uniq === 'number') {
    doc.setFontSize(12);
    doc.setTextColor(0);
    doc.text(`Unique Counterparties: ${uniq}`, 20, 108);
  }

  // Helper to draw a simple line chart (cumulative BTC) and a bar overlay (tx_count)
  const drawChart = (x: number, y: number, width: number, height: number, title: string, data?: WalletTimeSeriesResponse | null) => {
    doc.setDrawColor(200);
    doc.rect(x, y, width, height);
    doc.setFontSize(12);
    doc.setTextColor(0);
    doc.text(title, x + 4, y - 4);
    if (!data || !data.points || data.points.length === 0) {
      doc.setFontSize(10);
      doc.setTextColor(120);
      doc.text('No data available', x + width / 2, y + height / 2, { align: 'center' });
      return;
    }

    const pts = data.points;
    const pad = 8;
    const gx = x + pad;
    const gy = y + pad;
    const gw = width - pad * 2;
    const gh = height - pad * 2;

    // Scales
    const cum = pts.map(p => p.cumulative_balance_btc);
    const txs = pts.map(p => p.tx_count);
    const minCum = Math.min(...cum, 0);
    const maxCum = Math.max(...cum, 0.00000001);
    const maxTx = Math.max(...txs, 1);

    const toX = (i: number) => gx + (i / Math.max(pts.length - 1, 1)) * gw;
    const toYCum = (v: number) => gy + gh - ((v - minCum) / (maxCum - minCum)) * gh;
    const toYTx = (v: number) => gy + gh - (v / maxTx) * gh * 0.5; // bars take up half area

    // Bars for tx_count
    doc.setDrawColor(96, 165, 250);
    doc.setFillColor(96, 165, 250);
    const barWidth = Math.max(1, gw / Math.max(pts.length, 1) * 0.6);
    pts.forEach((p, i) => {
      const bx = toX(i) - barWidth / 2;
      const by = toYTx(p.tx_count);
      const bh = gy + gh - by;
      doc.rect(bx, by, barWidth, bh, 'F');
    });

    // Line for cumulative BTC
    doc.setDrawColor(139, 92, 246);
    doc.setLineWidth(0.6);
    pts.forEach((p, i) => {
      const cx = toX(i);
      const cy = toYCum(p.cumulative_balance_btc);
      if (i === 0) doc.moveTo(cx, cy);
      else doc.lineTo(cx, cy);
    });
    doc.stroke();
  };

  // Layout two charts per row
  const chartWidth = 85;
  const chartHeight = 55;

  // Weekly chart
  drawChart(20, 115, chartWidth, chartHeight, 'Weekly Activity (Cumulative BTC & Tx Count)', weekly);

  // Monthly chart
  drawChart(20 + chartWidth + 15, 115, chartWidth, chartHeight, 'Monthly Activity (Cumulative BTC & Tx Count)', monthly);

  // Finalize with optional sections reused from original
  let startY = 115 + chartHeight + 20;
  // Risk factors
  if (analysis.risk_factors && analysis.risk_factors.length > 0) {
    doc.setFontSize(16);
    doc.setTextColor(0);
    doc.text('Risk Factors', 20, startY);
    (doc as any).autoTable({
      startY: startY + 5,
      head: [['#', 'Risk Factor']],
      body: analysis.risk_factors.map((factor, idx) => [`${idx + 1}.`, factor]),
      theme: 'grid', styles: { fontSize: 10 }, headStyles: { fillColor: [37, 99, 235] }, margin: { horizontal: 20 }
    });
    startY = (doc as any).lastAutoTable?.finalY + 10 || startY + 25;
  }

  // Positive indicators
  if (analysis.positive_indicators && analysis.positive_indicators.length > 0) {
    doc.setFontSize(16);
    doc.text('Positive Indicators', 20, startY);
    (doc as any).autoTable({
      startY: startY + 5,
      head: [['#', 'Positive Indicator']],
      body: analysis.positive_indicators.map((ind, idx) => [`${idx + 1}.`, ind]),
      theme: 'grid', styles: { fontSize: 10 }, headStyles: { fillColor: [5, 150, 105] }, margin: { horizontal: 20 }
    });
    startY = (doc as any).lastAutoTable?.finalY + 10 || startY + 25;
  }

  // Data limitations
  if (analysis.data_limitations) {
    doc.setFontSize(16);
    doc.text('Data Limitations', 20, startY);
    (doc as any).autoTable({
      startY: startY + 5,
      head: [['Limitation', 'Value']],
      body: [
        ['Rate Limit Detected', analysis.data_limitations.rate_limit_detected ? 'Yes' : 'No'],
        ['Real Time Data', analysis.data_limitations.real_time_data ? 'Yes' : 'No'],
        ['API Status', analysis.data_limitations.api_status],
        ['Note', analysis.data_limitations.note || 'N/A'],
        ['Description', analysis.data_limitations.description || 'N/A'],
        ['Accuracy Note', analysis.data_limitations.accuracy_note || 'N/A'],
        ['Recommendation', analysis.data_limitations.recommendation || 'N/A']
      ],
      theme: 'grid', styles: { fontSize: 10 }, headStyles: { fillColor: [245, 158, 11] }, margin: { horizontal: 20 }
    });
  }

  // Save file
  doc.save(`BitScan_Report_${address.substring(0, 8)}_${new Date().toISOString().slice(0, 10)}.pdf`);
};