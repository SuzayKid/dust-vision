import { useEffect, useState, useRef } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  Camera,
  Plus,
  Download,
  Activity,
  Wind,
  Server,
  AlertTriangle,
  MapPin,
  X,
  CheckCircle2,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

// --- Utility for cleaner tailwind classes ---
function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// --- Constants & Styles ---
const STATUS_CONFIG = {
  good: { color: "#10B981", label: "Good", bg: "bg-emerald-500/20", border: "border-emerald-500/50", text: "text-emerald-400" },
  moderate: { color: "#F59E0B", label: "Moderate", bg: "bg-amber-500/20", border: "border-amber-500/50", text: "text-amber-400" },
  unhealthy: { color: "#EF4444", label: "Unhealthy", bg: "bg-red-500/20", border: "border-red-500/50", text: "text-red-400" },
  hazardous: { color: "#7F1D1D", label: "Hazardous", bg: "bg-rose-900/40", border: "border-rose-500", text: "text-rose-500" },
};

const INITIAL_DATA = {
  system_status: "OPTIMAL",
  timestamp: new Date().toISOString(),
  nodes: [
    {
      id: "DV-NODE-01",
      location: "North Gate Sector",
      pm25: 45.2,
      pm10: 180.5,
      predicted_pm10: 265.2,
      camera_detection: { detected: true, severity: 0.8 },
      status: "unhealthy",
    },
    {
      id: "DV-NODE-02",
      location: "East Construction Zone",
      pm25: 12.0,
      pm10: 35.5,
      predicted_pm10: 38.0,
      camera_detection: { detected: false, severity: 0 },
      status: "good",
    },
  ],
  recent_logs: [
    { id: 1, time: "10:29:45", type: "ALERT", message: "High PM10 forecast (265 µg/m³) at Node 1" },
    { id: 2, time: "10:29:50", type: "ACTION", message: "Auto-Triggered Sprinklers (Intensity: 90%)" },
  ],
};

// --- Helper Functions ---
function getStatus(pm10) {
  if (pm10 < 50) return "good";
  if (pm10 < 150) return "moderate";
  if (pm10 < 250) return "unhealthy";
  return "hazardous";
}

// --- Components ---

// 1. Glass Card Wrapper
const GlassCard = ({ children, className, hover = false }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    whileHover={hover ? { scale: 1.01, backgroundColor: "rgba(255,255,255,0.07)" } : {}}
    className={cn(
      "backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl p-5 shadow-xl transition-colors",
      className
    )}
  >
    {children}
  </motion.div>
);

// 2. Custom Chart Tooltip
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900/90 border border-slate-700 p-3 rounded-lg shadow-2xl backdrop-blur-md">
        <p className="text-slate-400 text-xs mb-1">{label}</p>
        <p className="text-emerald-400 text-sm font-mono">
          Actual: <span className="font-bold">{payload[0].value.toFixed(1)}</span>
        </p>
        <p className="text-cyan-400 text-sm font-mono">
          AI Forecast: <span className="font-bold">{payload[1].value.toFixed(1)}</span>
        </p>
      </div>
    );
  }
  return null;
};

// 3. Stat Card
const StatCard = ({ title, value, icon: Icon, trend }) => (
  <GlassCard hover className="flex flex-col justify-between h-32 relative overflow-hidden group">
    <div className="absolute -right-4 -top-4 bg-gradient-to-br from-cyan-500/20 to-transparent w-24 h-24 rounded-full blur-2xl group-hover:bg-cyan-500/30 transition-all" />
    <div className="flex justify-between items-start z-10">
      <div>
        <p className="text-slate-400 text-sm font-medium uppercase tracking-wider">{title}</p>
        <h3 className="text-3xl font-bold text-white mt-2 tracking-tight">{value}</h3>
      </div>
      <div className="p-2 bg-white/5 rounded-lg text-cyan-400">
        <Icon size={20} />
      </div>
    </div>
    {trend && <p className="text-xs text-slate-500 mt-2 font-mono">{trend}</p>}
  </GlassCard>
);

// --- Main Application ---
export default function App() {
  const [data, setData] = useState(INITIAL_DATA);
  const [chartData, setChartData] = useState([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newNode, setNewNode] = useState({ id: "", location: "" });
  const logsEndRef = useRef(null);

  // // Auto-scroll logs
  // useEffect(() => {
  //   logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  // }, [data.recent_logs]);

  // Simulation Loop
  useEffect(() => {
    const interval = setInterval(() => {
      setData((prev) => {
        const updatedNodes = prev.nodes.map((n) => {
          // Simulate fluctuation
          const delta = (Math.random() - 0.5) * 15;
          const pm10 = Math.max(10, n.pm10 + delta);
          const predicted = pm10 + 40 + Math.random() * 20;
          const newStatus = getStatus(pm10);
          
          return {
            ...n,
            pm10,
            predicted_pm10: predicted,
            status: newStatus,
            camera_detection: { ...n.camera_detection, detected: pm10 > 200 },
          };
        });

        // Generate logs if hazardous
        let newLogs = [...prev.recent_logs];
        const hazardousNode = updatedNodes.find((n) => n.pm10 > 250);
        if (hazardousNode && Math.random() > 0.7) {
            const id = Date.now();
            newLogs.push({
                id,
                time: new Date().toLocaleTimeString(),
                type: "ALERT",
                message: `Critical PM10 Spike detected at ${hazardousNode.id}`
            });
            if(newLogs.length > 50) newLogs.shift();
        }

        const timeLabel = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        setChartData((cd) => {
          const newEntry = {
            time: timeLabel,
            pm10: updatedNodes[0].pm10,
            forecast: updatedNodes[0].predicted_pm10,
          };
          const newData = [...cd, newEntry];
          return newData.slice(-40); // Keep last 40 points
        });

        return {
          ...prev,
          timestamp: new Date().toISOString(),
          nodes: updatedNodes,
          recent_logs: newLogs
        };
      });
    }, 1500); // Faster update rate for "fluidity"
    return () => clearInterval(interval);
  }, []);

  const handleAddSensor = () => {
    if(!newNode.id || !newNode.location) return;
    setData((d) => ({
      ...d,
      nodes: [
        ...d.nodes,
        {
          id: newNode.id,
          location: newNode.location,
          pm25: 15,
          pm10: 30,
          predicted_pm10: 45,
          camera_detection: { detected: false },
          status: "good",
        },
      ],
    }));
    setIsModalOpen(false);
    setNewNode({ id: "", location: "" });
  };

  return (
    <div className="min-h-screen bg-[#0B0F19] text-slate-200 font-sans selection:bg-cyan-500/30">
      {/* Background Gradients */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-cyan-900/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-900/20 rounded-full blur-[120px]" />
      </div>

      <div className="relative z-10 flex h-screen overflow-hidden">
        {/* SIDEBAR */}
        <aside className="w-64 bg-slate-900/50 backdrop-blur-xl border-r border-white/5 flex flex-col p-6 gap-8">
          <div className="flex items-center gap-3 px-2">
            <div className="w-8 h-8 bg-gradient-to-tr from-cyan-400 to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Activity className="text-white" size={18} />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white">
              Dust<span className="text-cyan-400">Vision</span>
            </h1>
          </div>

          <nav className="flex flex-col gap-2">
            {['Dashboard', 'Analytics', 'Map View', 'Compliance'].map((item, i) => (
              <button key={item} className={cn(
                "flex items-center gap-3 px-4 py-3 rounded-xl transition-all",
                i === 0 ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20" : "hover:bg-white/5 text-slate-400 hover:text-white"
              )}>
                <div className={cn("w-1.5 h-1.5 rounded-full", i === 0 ? "bg-cyan-400" : "bg-transparent")} />
                {item}
              </button>
            ))}
          </nav>

          <div className="mt-auto">
             <button
              onClick={() => setIsModalOpen(true)}
              className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium py-3 rounded-xl shadow-lg shadow-cyan-900/20 transition-all active:scale-95"
            >
              <Plus size={18} /> Deploy Sensor
            </button>
          </div>
        </aside>

        {/* MAIN CONTENT */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Header */}
          <header className="h-20 border-b border-white/5 flex items-center justify-between px-8 bg-slate-900/30 backdrop-blur-sm">
            <div>
                <h2 className="text-xl font-semibold text-white">Real-time Monitoring</h2>
                <p className="text-xs text-slate-500 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"/> 
                    System Operational • Last Sync: {new Date().toLocaleTimeString()}
                </p>
            </div>
            <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800/50 hover:bg-slate-700/50 border border-white/10 text-sm transition-colors">
              <Download size={16} /> Export Data
            </button>
          </header>

          {/* Scrollable Dashboard Area */}
          <div className="flex-1 overflow-y-auto p-8 space-y-6 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
            
            {/* Top Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <StatCard title="Avg PM10 Level" value={Math.round(data.nodes.reduce((a, b) => a + b.pm10, 0) / data.nodes.length)} icon={Wind} trend="Running avg over all nodes" />
              <StatCard title="Active Nodes" value={data.nodes.length} icon={Server} trend="100% Uptime" />
              <StatCard title="Alerts (24h)" value={data.recent_logs.filter(l => l.type === 'ALERT').length} icon={AlertTriangle} trend="Requires attention" />
              <StatCard title="Mitigation Rate" value="94%" icon={CheckCircle2} trend="Automatic trigger success" />
            </div>

            {/* Main Grid: Charts & Feeds */}
            <div className="grid grid-cols-12 gap-6 h-[500px]">
              
              {/* Main Chart */}
              <GlassCard className="col-span-8 flex flex-col">
                <div className="flex justify-between items-center mb-6">
                    <div>
                        <h3 className="text-lg font-semibold text-white">Pollution Forecast Engine</h3>
                        <p className="text-sm text-slate-400">Real-time sensor data vs. AI Prediction Model</p>
                    </div>
                    <div className="flex gap-4 text-xs">
                        <span className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-cyan-400"/> Forecast</span>
                        <span className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-emerald-500"/> Actual</span>
                    </div>
                </div>
                <div className="flex-1 w-full min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="colorPm" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="colorFc" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#22D3EE" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#22D3EE" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} opacity={0.3} />
                      <XAxis dataKey="time" stroke="#64748B" fontSize={11} tickLine={false} axisLine={false} />
                      <YAxis stroke="#64748B" fontSize={11} tickLine={false} axisLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine y={250} stroke="#EF4444" strokeDasharray="3 3" label={{ value: 'HAZARDOUS', fill: '#EF4444', fontSize: 10, position: 'insideTopRight' }} />
                      <Area type="monotone" dataKey="forecast" stroke="#22D3EE" strokeWidth={2} strokeDasharray="5 5" fill="url(#colorFc)" animationDuration={1000} />
                      <Area type="monotone" dataKey="pm10" stroke="#10B981" strokeWidth={2} fill="url(#colorPm)" animationDuration={1000} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>

              {/* Node Feed */}
              <GlassCard className="col-span-4 flex flex-col overflow-hidden">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold text-white">Live Node Feed</h3>
                    <span className="text-xs px-2 py-1 rounded-full bg-slate-800 text-slate-400">{data.nodes.length} Online</span>
                </div>
                <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-slate-700">
                  <AnimatePresence>
                    {data.nodes.map((node) => {
                      const status = STATUS_CONFIG[node.status];
                      return (
                        <motion.div
                          key={node.id}
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          className={cn(
                            "p-4 rounded-xl border transition-all relative overflow-hidden group",
                            status.bg,
                            status.border
                          )}
                        >
                          {/* Glow effect for unhealthy nodes */}
                          {node.status === 'hazardous' || node.status === 'unhealthy' ? (
                             <div className="absolute inset-0 bg-red-500/10 animate-pulse pointer-events-none" />
                          ) : null}

                          <div className="flex justify-between items-start relative z-10">
                            <div className="flex items-start gap-3">
                                <div className={cn("p-2 rounded-lg bg-black/20", status.text)}>
                                    <MapPin size={18} />
                                </div>
                                <div>
                                    <h4 className="font-bold text-white text-sm">{node.id}</h4>
                                    <p className="text-xs text-slate-300 opacity-80">{node.location}</p>
                                </div>
                            </div>
                            {node.camera_detection.detected && (
                                <div className="flex items-center gap-1 text-[10px] font-bold bg-red-500 text-white px-2 py-1 rounded animate-pulse">
                                    <Camera size={10} /> VISUAL
                                </div>
                            )}
                          </div>
                          
                          <div className="mt-3 flex items-end justify-between relative z-10">
                             <div>
                                <p className="text-[10px] uppercase text-slate-400 font-semibold">PM10 Level</p>
                                <p className="text-2xl font-bold text-white leading-none">{node.pm10.toFixed(1)}</p>
                             </div>
                             <div className={cn("text-xs font-bold px-2 py-1 rounded border", status.text, status.border)}>
                                {status.label}
                             </div>
                          </div>
                        </motion.div>
                      );
                    })}
                  </AnimatePresence>
                </div>
              </GlassCard>
            </div>

            {/* System Logs */}
            <GlassCard className="h-64 flex flex-col">
                <h3 className="text-sm font-semibold text-slate-400 mb-3 uppercase tracking-wider flex items-center gap-2">
                    <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-pulse" /> System Logs
                </h3>
                <div className="flex-1 overflow-y-auto font-mono text-xs space-y-2 pr-2 scrollbar-thin scrollbar-thumb-slate-700">
                    <AnimatePresence>
                        {data.recent_logs.map((log) => (
                            <motion.div 
                                key={log.id} 
                                initial={{ opacity: 0, height: 0 }} 
                                animate={{ opacity: 1, height: 'auto' }}
                                className="flex gap-3 border-b border-white/5 pb-2 last:border-0"
                            >
                                <span className="text-slate-500 min-w-[80px]">{log.time}</span>
                                <span className={cn(
                                    "font-bold min-w-[60px]",
                                    log.type === 'ALERT' ? 'text-red-400' : 
                                    log.type === 'ACTION' ? 'text-cyan-400' : 'text-slate-400'
                                )}>{log.type}</span>
                                <span className="text-slate-300">{log.message}</span>
                            </motion.div>
                        ))}
                        <div ref={logsEndRef} />
                    </AnimatePresence>
                </div>
            </GlassCard>

          </div>
        </main>
      </div>

      {/* MODAL */}
      <AnimatePresence>
        {isModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <motion.div 
                initial={{ opacity: 0 }} 
                animate={{ opacity: 1 }} 
                exit={{ opacity: 0 }}
                className="absolute inset-0 bg-black/80 backdrop-blur-sm" 
                onClick={() => setIsModalOpen(false)}
            />
            <motion.div
                initial={{ scale: 0.9, opacity: 0 }} 
                animate={{ scale: 1, opacity: 1 }} 
                exit={{ scale: 0.9, opacity: 0 }}
                className="relative bg-slate-900 border border-white/10 p-6 rounded-2xl w-full max-w-md shadow-2xl"
            >
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-bold text-white">Deploy New Sensor</h3>
                    <button onClick={() => setIsModalOpen(false)} className="text-slate-400 hover:text-white"><X size={20}/></button>
                </div>
                
                <div className="space-y-4">
                    <div>
                        <label className="block text-xs uppercase text-slate-500 font-bold mb-2">Node Identifier</label>
                        <input 
                            value={newNode.id}
                            onChange={(e) => setNewNode({...newNode, id: e.target.value})}
                            placeholder="e.g. DV-NODE-03" 
                            className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-white focus:ring-2 focus:ring-cyan-500 outline-none transition-all"
                        />
                    </div>
                    <div>
                        <label className="block text-xs uppercase text-slate-500 font-bold mb-2">Physical Location</label>
                        <input 
                            value={newNode.location}
                            onChange={(e) => setNewNode({...newNode, location: e.target.value})}
                            placeholder="e.g. South Perimeter" 
                            className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-white focus:ring-2 focus:ring-cyan-500 outline-none transition-all"
                        />
                    </div>
                    <button 
                        onClick={handleAddSensor}
                        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-bold py-3 rounded-lg mt-2 transition-colors"
                    >
                        Activate Node
                    </button>
                </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}