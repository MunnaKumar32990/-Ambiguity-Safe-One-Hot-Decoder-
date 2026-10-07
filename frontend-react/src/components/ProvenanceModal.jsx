import React, { useState } from 'react';
import { X, Copy, Check, ShieldCheck, AlertTriangle } from 'lucide-react';

export default function ProvenanceModal({ sample, onClose }) {
  const [copied, setCopied] = useState(false);

  if (!sample) return null;

  const copyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(sample.provenance, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isAmbiguous = sample.row_status === 'AMBIGUOUS';

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '20px'
    }}>
      {/* Backdrop */}
      <div 
        onClick={onClose}
        style={{
          position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(6px)'
        }}
      />

      {/* Modal Card */}
      <div className="glass-panel" style={{
        position: 'relative', width: '100%', maxWidth: '640px',
        maxHeight: '85vh', display: 'flex', flexDirection: 'column',
        zIndex: 101, overflow: 'hidden',
        boxShadow: 'var(--shadow-lg)'
      }}>
        {/* Modal Header */}
        <div style={{
          padding: '18px 22px', borderBottom: '1px solid var(--border-subtle)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: 'var(--bg-surface-elevated)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {isAmbiguous ? (
              <div style={{ color: 'var(--ambiguous)', display: 'flex' }}><AlertTriangle size={20} /></div>
            ) : (
              <div style={{ color: 'var(--safe)', display: 'flex' }}><ShieldCheck size={20} /></div>
            )}
            <div>
              <h3 style={{ fontSize: '16px', fontWeight: 700 }}>
                Sample #{sample.sample_index + 1} Provenance Side-Channel Audit Trace
              </h3>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                Policy Applied: <strong style={{ color: 'var(--primary-light)' }}>{sample.policy_applied}</strong> &middot; Status: <strong style={{ color: isAmbiguous ? 'var(--ambiguous)' : 'var(--safe)' }}>{sample.row_status}</strong>
              </p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="btn btn-secondary" 
            style={{ width: '32px', height: '32px', padding: 0 }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '20px 22px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          {/* Quick Metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px' }}>
            <div style={{ padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-surface-subtle)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Confidence</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: sample.confidence === 'HIGH' ? 'var(--safe)' : 'var(--ambiguous)' }}>
                {sample.confidence}
              </div>
            </div>
            <div style={{ padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-surface-subtle)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Collision Status</div>
              <div style={{ fontSize: '14px', fontWeight: 700, color: isAmbiguous ? 'var(--ambiguous)' : 'var(--safe)' }}>
                {isAmbiguous ? 'COLLISION DETECTED' : 'UNAMBIGUOUS'}
              </div>
            </div>
            <div style={{ padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-surface-subtle)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Candidates</div>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                {sample.possible_values.map(p => p.join('|')).join(', ')}
              </div>
            </div>
          </div>

          {/* JSON Viewer */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                Structured Provenance Payload
              </span>
              <button 
                onClick={copyJson}
                className="btn btn-secondary btn-sm"
                style={{ gap: '6px' }}
              >
                {copied ? <Check size={13} style={{ color: 'var(--safe)' }} /> : <Copy size={13} />}
                <span>{copied ? 'Copied!' : 'Copy JSON'}</span>
              </button>
            </div>
            <pre style={{
              background: 'var(--bg-page)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              padding: '14px',
              fontFamily: 'var(--font-mono)',
              fontSize: '12px',
              color: 'var(--text-primary)',
              overflowX: 'auto',
              maxHeight: '300px'
            }}>
              {JSON.stringify(sample.provenance, null, 2)}
            </pre>
          </div>

        </div>

        {/* Modal Footer */}
        <div style={{
          padding: '12px 22px', borderTop: '1px solid var(--border-subtle)',
          display: 'flex', justifyContent: 'flex-end', background: 'var(--bg-surface-elevated)'
        }}>
          <button onClick={onClose} className="btn btn-primary btn-sm">Close Audit Trace</button>
        </div>
      </div>
    </div>
  );
}
