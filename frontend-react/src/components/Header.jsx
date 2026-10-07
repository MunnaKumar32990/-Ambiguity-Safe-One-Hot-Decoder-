import React from 'react';
import { Shield, Moon, Sun, Activity, CheckCircle2, AlertCircle } from 'lucide-react';

export default function Header({ theme, toggleTheme, apiHealth }) {
  return (
    <header className="glass-panel" style={{ borderRadius: 0, borderTop: 0, borderLeft: 0, borderRight: 0, position: 'sticky', top: 0, zIndex: 40 }}>
      <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '14px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px' }}>
        
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ 
            width: '42px', height: '42px', borderRadius: '12px', 
            background: 'linear-gradient(135deg, var(--primary) 0%, #1e40af 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 4px 12px var(--primary-glow)',
            color: '#fff'
          }}>
            <Shield size={24} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
              <h1 style={{ fontSize: '18px', fontWeight: 800 }}>Ambiguity-Safe Inverse Decoding</h1>
              <span className="badge badge-code">KLCAP-2026-00332</span>
              <span className="badge badge-safe">CP1 Gate 2 Ready</span>
            </div>
            <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              scikit-learn Issue #34549 Resolution &middot; Explicit Ambiguity Policies &middot; Verification Harness
            </p>
          </div>
        </div>

        {/* Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {/* Health Pill */}
          <div style={{ 
            display: 'flex', alignItems: 'center', gap: '6px', 
            padding: '6px 12px', borderRadius: 'var(--radius-full)', 
            background: apiHealth.ok ? 'var(--safe-tint)' : 'var(--ambiguous-tint)',
            color: apiHealth.ok ? 'var(--safe)' : 'var(--ambiguous)',
            fontSize: '12px', fontWeight: 600, border: `1px solid ${apiHealth.ok ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)'}`
          }}>
            {apiHealth.ok ? <CheckCircle2 size={14} /> : <AlertCircle size={14} />}
            <span>{apiHealth.text}</span>
          </div>

          {/* Theme Toggle */}
          <button 
            onClick={toggleTheme}
            className="btn btn-secondary"
            style={{ width: '38px', height: '38px', padding: 0 }}
            title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? <Sun size={17} style={{ color: '#fbbf24' }} /> : <Moon size={17} style={{ color: '#6366f1' }} />}
          </button>
        </div>

      </div>
    </header>
  );
}
