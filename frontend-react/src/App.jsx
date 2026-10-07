import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import TabNav from './components/TabNav';
import SandboxTab from './components/SandboxTab';
import NegativeTestsTab from './components/NegativeTestsTab';
import KpisTab from './components/KpisTab';
import ProvenanceModal from './components/ProvenanceModal';

export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark';
  });

  const [activeTab, setActiveTab] = useState('sandbox');
  const [apiHealth, setApiHealth] = useState({ ok: true, text: 'Connecting to API…' });
  const [provenanceSample, setProvenanceSample] = useState(null);

  // Sync theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Health check
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => {
        setApiHealth({ 
          ok: true, 
          text: `API ONLINE · scikit-learn v${data.sklearn_version}` 
        });
      })
      .catch(() => {
        setApiHealth({ 
          ok: false, 
          text: 'API OFFLINE (Run python run_app.py)' 
        });
      });
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Topbar */}
      <Header theme={theme} toggleTheme={toggleTheme} apiHealth={apiHealth} />

      {/* Tabs */}
      <TabNav activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Content Area */}
      <main style={{ maxWidth: '1440px', width: '100%', margin: '24px auto', padding: '0 24px', flex: 1 }}>
        {activeTab === 'sandbox' && (
          <SandboxTab onOpenProvenance={(sample) => setProvenanceSample(sample)} />
        )}
        {activeTab === 'negative-tests' && (
          <NegativeTestsTab />
        )}
        {activeTab === 'kpis' && (
          <KpisTab />
        )}
      </main>

      {/* Provenance Audit Modal */}
      {provenanceSample && (
        <ProvenanceModal 
          sample={provenanceSample} 
          onClose={() => setProvenanceSample(null)} 
        />
      )}

      {/* Footer */}
      <footer style={{
        background: 'var(--bg-surface-elevated)',
        borderTop: '1px solid var(--border-subtle)',
        padding: '16px 24px', marginTop: '40px'
      }}>
        <div style={{
          maxWidth: '1440px', margin: '0 auto',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          fontSize: '12px', color: 'var(--text-muted)', flexWrap: 'wrap', gap: '10px'
        }}>
          <div>
            <strong>KLCAP-2026-00332</strong> &middot; Ambiguity-Safe Inverse Decoding for One-Hot Encoded Categories
          </div>
          <div>
            B.Tech 4th Year (Odd Semester) Capstone Review-2 &middot; React + Flask Architecture
          </div>
        </div>
      </footer>

    </div>
  );
}
