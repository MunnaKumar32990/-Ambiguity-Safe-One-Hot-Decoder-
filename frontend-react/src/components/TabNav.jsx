import React from 'react';
import { FlaskConical, ShieldAlert, BarChart3, FileCheck2 } from 'lucide-react';

export default function TabNav({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'sandbox', label: 'Interactive Sandbox & Policies', icon: FlaskConical },
    { id: 'negative-tests', label: 'Mandatory Negative Tests (NT-1..5)', icon: ShieldAlert, badge: '5 Tests' },
    { id: 'kpis', label: 'KPI Benchmarks (KPI-1..6)', icon: BarChart3, badge: '100% Pass' },
    { id: 'evidence', label: 'Contract & Deliverables (D1..D7)', icon: FileCheck2 },
  ];

  return (
    <nav style={{ 
      background: 'var(--bg-surface-elevated)', 
      borderBottom: '1px solid var(--border-subtle)',
      position: 'sticky', top: '70px', zIndex: 30
    }}>
      <div style={{ maxWidth: '1440px', margin: '0 auto', padding: '0 24px', display: 'flex', gap: '8px', overflowX: 'auto' }}>
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '9px',
                padding: '13px 18px',
                background: 'none',
                border: 'none',
                borderBottom: `2px solid ${isActive ? 'var(--primary)' : 'transparent'}`,
                color: isActive ? 'var(--primary)' : 'var(--text-secondary)',
                fontWeight: isActive ? 700 : 500,
                fontSize: '13.5px',
                fontFamily: 'inherit',
                cursor: 'pointer',
                transition: 'all 0.15s',
                whiteSpace: 'nowrap'
              }}
            >
              <Icon size={16} style={{ color: isActive ? 'var(--primary)' : 'var(--text-muted)' }} />
              <span>{tab.label}</span>
              {tab.badge && (
                <span style={{
                  fontSize: '10.5px',
                  fontWeight: 700,
                  padding: '2px 6px',
                  borderRadius: 'var(--radius-full)',
                  background: isActive ? 'var(--primary-tint)' : 'var(--bg-surface-subtle)',
                  color: isActive ? 'var(--primary-light)' : 'var(--text-muted)',
                  border: `1px solid ${isActive ? 'rgba(59,130,246,0.3)' : 'transparent'}`
                }}>
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
}
