import { useState, useCallback, useEffect, useRef } from 'react';
import { useIngestion } from '../hooks/useApi';
import type { IngestionJob, IngestionStage } from '../hooks/useApi';

const SUPPORTED_FORMATS = [
    { ext: '.pdf', label: 'PDF' },
    { ext: '.docx', label: 'Word' },
    { ext: '.xlsx', label: 'Excel' },
    { ext: '.csv', label: 'CSV' },
    { ext: '.txt', label: 'Text' },
    { ext: '.md', label: 'Markdown' },
    { ext: '.png', label: 'PNG' },
    { ext: '.jpg', label: 'JPEG' },
    { ext: '.jpeg', label: 'JPEG' },
    { ext: '.webp', label: 'WebP' },
];

function StageIndicator({ stage }: { stage: IngestionStage }) {
    const getStatusIcon = () => {
        switch (stage.status) {
            case 'completed':
                return (
                    <span style={{ color: 'var(--color-success)' }}>✓</span>
                );
            case 'running':
                return (
                    <span className="spinner" style={{ width: 14, height: 14 }} />
                );
            case 'failed':
                return (
                    <span style={{ color: 'var(--color-error)' }}>✗</span>
                );
            default:
                return (
                    <span style={{ color: 'var(--color-text-muted)' }}>○</span>
                );
        }
    };

    const getStatusColor = () => {
        switch (stage.status) {
            case 'completed':
                return 'var(--color-success)';
            case 'running':
                return 'var(--color-accent-primary)';
            case 'failed':
                return 'var(--color-error)';
            default:
                return 'var(--color-text-muted)';
        }
    };

    return (
        <div
            style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-2)',
                padding: 'var(--space-2) var(--space-3)',
                background: stage.status === 'running' ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
                borderRadius: 'var(--radius-md)',
                borderLeft: `3px solid ${getStatusColor()}`,
            }}
        >
            <div style={{ width: 18, display: 'flex', justifyContent: 'center' }}>
                {getStatusIcon()}
            </div>
            <div style={{ flex: 1 }}>
                <div style={{ fontSize: 'var(--text-sm)', fontWeight: 500 }}>
                    {stage.name}
                </div>
                {stage.items_processed > 0 && (
                    <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)' }}>
                        {stage.items_processed} items
                    </div>
                )}
                {stage.error && (
                    <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-error)' }}>
                        {stage.error}
                    </div>
                )}
            </div>
        </div>
    );
}

function JobCard({ job, onRefresh }: { job: IngestionJob; onRefresh: () => void }) {
    const getStatusBadge = () => {
        const badges: Record<string, { color: string; bg: string; label: string }> = {
            pending: { color: 'var(--color-warning)', bg: 'rgba(245, 158, 11, 0.2)', label: 'Pending' },
            processing: { color: 'var(--color-info)', bg: 'rgba(59, 130, 246, 0.2)', label: 'Processing' },
            completed: { color: 'var(--color-success)', bg: 'rgba(34, 197, 94, 0.2)', label: 'Completed' },
            failed: { color: 'var(--color-error)', bg: 'rgba(239, 68, 68, 0.2)', label: 'Failed' },
        };
        const badge = badges[job.status] || badges.pending;
        return (
            <span
                style={{
                    background: badge.bg,
                    color: badge.color,
                    padding: 'var(--space-1) var(--space-2)',
                    borderRadius: 'var(--radius-full)',
                    fontSize: 'var(--text-xs)',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                }}
            >
                {badge.label}
            </span>
        );
    };

    return (
        <div className="card" style={{ marginBottom: 'var(--space-4)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 'var(--space-4)' }}>
                <div>
                    <div style={{ fontSize: 'var(--text-lg)', fontWeight: 600, marginBottom: 'var(--space-1)' }}>
                        {job.file_name}
                    </div>
                    <div style={{ display: 'flex', gap: 'var(--space-3)', alignItems: 'center' }}>
                        <span className="badge badge-info">{job.file_type.toUpperCase()}</span>
                        {getStatusBadge()}
                    </div>
                </div>
                <button className="btn btn-icon" onClick={onRefresh} title="Refresh">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="23 4 23 10 17 10" />
                        <polyline points="1 20 1 14 7 14" />
                        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                    </svg>
                </button>
            </div>

            {/* Progress Bar */}
            <div style={{ marginBottom: 'var(--space-4)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--space-1)' }}>
                    <span style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-secondary)' }}>Progress</span>
                    <span style={{ fontSize: 'var(--text-sm)', fontWeight: 600 }}>{Math.round(job.progress * 100)}%</span>
                </div>
                <div className="progress-bar">
                    <div
                        className="progress-bar-fill"
                        style={{
                            width: `${job.progress * 100}%`,
                            background: job.status === 'failed' ? 'var(--color-error)' : undefined,
                        }}
                    />
                </div>
            </div>

            {/* Stages */}
            {job.stages.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
                    {job.stages.map((stage, idx) => (
                        <StageIndicator key={idx} stage={stage} />
                    ))}
                </div>
            )}

            {/* Error Message */}
            {job.error && (
                <div
                    style={{
                        marginTop: 'var(--space-4)',
                        padding: 'var(--space-3)',
                        background: 'rgba(239, 68, 68, 0.1)',
                        borderRadius: 'var(--radius-md)',
                        color: 'var(--color-error)',
                        fontSize: 'var(--text-sm)',
                    }}
                >
                    <strong>Error:</strong> {job.error}
                </div>
            )}

            {/* Result Summary */}
            {job.result && (
                <div
                    style={{
                        marginTop: 'var(--space-4)',
                        padding: 'var(--space-3)',
                        background: 'rgba(34, 197, 94, 0.1)',
                        borderRadius: 'var(--radius-md)',
                    }}
                >
                    <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, marginBottom: 'var(--space-2)', color: 'var(--color-success)' }}>
                        Ingestion Complete
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-2)' }}>
                        <div>
                            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)' }}>Pages</div>
                            <div style={{ fontWeight: 600 }}>{job.result.pages}</div>
                        </div>
                        <div>
                            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)' }}>OCR</div>
                            <div style={{ fontWeight: 600 }}>{job.result.pages_ocr}</div>
                        </div>
                        <div>
                            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)' }}>Chunks</div>
                            <div style={{ fontWeight: 600 }}>{job.result.child_chunks}</div>
                        </div>
                        <div>
                            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-tertiary)' }}>Entities</div>
                            <div style={{ fontWeight: 600 }}>{job.result.entities}</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export function FileUpload() {
    const { uploading, error: uploadError, uploadFile, getJobStatus, listJobs } = useIngestion();
    const [dragging, setDragging] = useState(false);
    const [jobs, setJobs] = useState<IngestionJob[]>([]);
    const [activeJobId, setActiveJobId] = useState<string | null>(null);
    const pollIntervalRef = useRef<number | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Load jobs on mount
    useEffect(() => {
        loadJobs();
    }, []);

    // Poll for active job status
    useEffect(() => {
        if (activeJobId) {
            pollIntervalRef.current = window.setInterval(async () => {
                const job = await getJobStatus(activeJobId);
                if (job) {
                    setJobs(prev => prev.map(j => j.job_id === job.job_id ? job : j));
                    
                    if (job.status === 'completed' || job.status === 'failed') {
                        setActiveJobId(null);
                        if (pollIntervalRef.current) {
                            clearInterval(pollIntervalRef.current);
                        }
                    }
                }
            }, 1000);

            return () => {
                if (pollIntervalRef.current) {
                    clearInterval(pollIntervalRef.current);
                }
            };
        }
    }, [activeJobId, getJobStatus]);

    const loadJobs = async () => {
        const jobList = await listJobs();
        setJobs(jobList);
    };

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragging(false);
    }, []);

    const handleDrop = useCallback(async (e: React.DragEvent) => {
        e.preventDefault();
        setDragging(false);

        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            await handleUpload(files[0]);
        }
    }, []);

    const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            await handleUpload(files[0]);
        }
        // Reset input
        e.target.value = '';
    }, []);

    const handleUpload = async (file: File) => {
        const result = await uploadFile(file);
        if (result) {
            const newJob: IngestionJob = {
                job_id: result.job_id,
                status: 'pending',
                file_name: file.name,
                file_type: file.name.split('.').pop() || 'unknown',
                progress: 0,
                stages: [],
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
            };
            setJobs(prev => [newJob, ...prev]);
            setActiveJobId(result.job_id);
        }
    };

    const refreshJob = async (jobId: string) => {
        const job = await getJobStatus(jobId);
        if (job) {
            setJobs(prev => prev.map(j => j.job_id === job.job_id ? job : j));
        }
    };

    return (
        <div className="animate-fadeIn">
            <div className="page-header">
                <h2>File Upload</h2>
                <p>Upload documents for multimodal ingestion with OCR support</p>
            </div>

            {/* Supported Formats */}
            <div style={{ marginBottom: 'var(--space-6)' }}>
                <div style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-2)' }}>
                    Supported formats:
                </div>
                <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
                    {SUPPORTED_FORMATS.map(f => (
                        <span
                            key={f.ext}
                            style={{
                                background: 'var(--color-bg-tertiary)',
                                padding: 'var(--space-1) var(--space-2)',
                                borderRadius: 'var(--radius-sm)',
                                fontSize: 'var(--text-xs)',
                                color: 'var(--color-text-secondary)',
                            }}
                        >
                            {f.ext}
                        </span>
                    ))}
                </div>
            </div>

            {/* Upload Zone */}
            <div
                className={`upload-zone ${dragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept={SUPPORTED_FORMATS.map(f => f.ext).join(',')}
                    onChange={handleFileSelect}
                    style={{ display: 'none' }}
                />

                {uploading ? (
                    <>
                        <div className="spinner" style={{ width: 48, height: 48, margin: '0 auto var(--space-4)' }} />
                        <p style={{ fontSize: 'var(--text-lg)', fontWeight: 500 }}>Uploading...</p>
                    </>
                ) : (
                    <>
                        <svg
                            className="upload-icon"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.5"
                        >
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                        <p style={{ fontSize: 'var(--text-lg)', fontWeight: 500, marginBottom: 'var(--space-2)' }}>
                            Drop files here or click to browse
                        </p>
                        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-tertiary)' }}>
                            PDF, DOCX, XLSX, CSV, TXT, MD, PNG, JPG, WEBP
                        </p>
                    </>
                )}
            </div>

            {/* Upload Error */}
            {uploadError && (
                <div
                    style={{
                        marginTop: 'var(--space-4)',
                        padding: 'var(--space-4)',
                        background: 'rgba(239, 68, 68, 0.1)',
                        borderRadius: 'var(--radius-lg)',
                        color: 'var(--color-error)',
                    }}
                >
                    <strong>Upload Error:</strong> {uploadError}
                </div>
            )}

            {/* Jobs List */}
            {jobs.length > 0 && (
                <div style={{ marginTop: 'var(--space-8)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
                        <h3>Ingestion Jobs</h3>
                        <button className="btn btn-secondary" onClick={loadJobs}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <polyline points="23 4 23 10 17 10" />
                                <polyline points="1 20 1 14 7 14" />
                                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                            </svg>
                            Refresh All
                        </button>
                    </div>
                    
                    {jobs.map(job => (
                        <JobCard key={job.job_id} job={job} onRefresh={() => refreshJob(job.job_id)} />
                    ))}
                </div>
            )}

            {/* Empty State */}
            {jobs.length === 0 && !uploading && (
                <div className="empty-state" style={{ marginTop: 'var(--space-8)' }}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                    </svg>
                    <p>No ingestion jobs yet. Upload a file to get started.</p>
                </div>
            )}
        </div>
    );
}
