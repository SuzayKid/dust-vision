import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, ResponsiveContainer } from "recharts";
import { Camera, Plus, Download } from "lucide-react";

const STATUS_COLORS = {
  good: "#00E400",
  moderate: "#FFFF00",
  unhealthy: "#FF0000",
  hazardous: "#7E0023",
};

const initialData = {
  system_status: "ONLINE",
  timestamp: new Date().toISOString(),
  nodes: [
    {
      id: "Node_1",
      location: "Site A - North Gate",
      pm25: 45.2,
      pm10: 180.5,
      predicted_pm10: 265.2,
      camera_detection: { detected: true, severity: 0.8, confidence: 0.95 },
      status: "unhealthy",
      mitigation_active: true,
    },
  ],
  recent_logs: [
    { time: "10:29:45", type: "ALERT", message: "High PM10 forecast (265 µg/m³) at Node 1" },
    { time: "10:29:50", type: "ACTION", message: "Triggered Sprinklers (Intensity: 90%)" },
  ],
};

function getStatus(pm10) {
  if (pm10 < 50) return "good";
  if (pm10 < 150) return "moderate";
  if (pm10 < 250) return "unhealthy";
  return "hazardous";
}

export default function App() {
  const [data, setData] = useState(initialData);
  const [chartData, setChartData] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [newNode, setNewNode] = useState({ id: "", location: "" });

  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => {
        const updatedNodes = prev.nodes.map(n => {
          const delta = (Math.random() - 0.5) * 20;
          const pm10 = Math.max(10, n.pm10 + delta);
          return {
            ...n,
            pm10,
            predicted_pm10: pm10 + 50 + Math.random() * 30,
            status: getStatus(pm10),
          };
        });

        const timeLabel = new Date().toLocaleTimeString();
        setChartData(cd => [
          ...cd.slice(-29),
          {
            time: timeLabel,
            pm10: updatedNodes[0].pm10,
            forecast: updatedNodes[0].predicted_pm10,
          },
        ]);

        return {
          ...prev,
          timestamp: new Date().toISOString(),
          nodes: updatedNodes,
        };
      });
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const addSensor = () => {
    setData(d => ({
      ...d,
      nodes: [
        ...d.nodes,
        {
          id: newNode.id,
          location: newNode.location,
          pm25: 20,
          pm10: 40,
          predicted_pm10: 70,
          camera_detection: { detected: false },
          status: "good",
        },
      ],
    }));
    setShowModal(false);
    setNewNode({ id: "", location: "" });
  };

  const exportCSV = () => {
    const headers = "time,pm10,forecast\n";
    const rows = chartData.map(r => `${r.time},${r.pm10.toFixed(2)},${r.forecast.toFixed(2)}`).join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "dustvision_pm10_history.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-screen grid grid-cols-[260px_1fr] bg-slate-950 text-slate-100">
      {/* Sidebar */}
      <aside className="bg-slate-900 p-6 flex flex-col gap-6">
        <h1 className="text-xl font-bold">DustVision AI</h1>
        <nav className="flex flex-col gap-3 text-slate-300">
          <span className="font-medium text-white">Dashboard</span>
          <span>Sensor Network</span>
          <span>Compliance Reports</span>
          <span>Settings</span>
        </nav>
        <button
          onClick={() => setShowModal(true)}
          className="mt-auto flex items-center gap-2 bg-cyan-600 hover:bg-cyan-700 px-4 py-2 rounded"
        >
          <Plus size={16} /> Add Sensor
        </button>
      </aside>

      {/* Dashboard Grid */}
      <div className="grid grid-rows-[auto_1fr_220px] gap-4 p-6 overflow-hidden">
        {/* Header */}
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-semibold">City Air Quality Dashboard</h2>
          <button
            onClick={exportCSV}
            className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded"
          >
            <Download size={16} /> Export CSV
          </button>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-[1fr_420px] gap-4 overflow-hidden">
          {/* Left Column */}
          <div className="grid grid-rows-[auto_1fr] gap-4">
            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
              <Stat title="System Health" value={data.system_status} />
              <Stat title="Active Sensors" value={data.nodes.length} />
              <Stat title="Avg PM10" value={Math.round(data.nodes.reduce((a, b) => a + b.pm10, 0) / data.nodes.length)} />
              <Stat title="Active Alerts" value={data.recent_logs.filter(l => l.type === "ALERT").length} />
            </div>

            {/* Chart */}
            <div className="bg-slate-900 rounded p-4">
              <h3 className="mb-2">PM10 – Real Time vs AI Forecast</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <ReferenceLine y={250} stroke="#FF0000" strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="pm10" strokeWidth={2} />
                  <Line type="monotone" dataKey="forecast" strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Sensor Column */}
          <div className="bg-slate-900 rounded p-4 overflow-y-auto">
            <h3 className="mb-3">Live Sensor Feed</h3>
            <div className="flex flex-col gap-3">
              {data.nodes.map(node => (
                <div
                  key={node.id}
                  className="p-3 rounded border"
                  style={{ borderColor: STATUS_COLORS[node.status] }}
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <p className="font-semibold">{node.id}</p>
                      <p className="text-xs text-slate-400">{node.location}</p>
                    </div>
                    <Camera
                      className={node.camera_detection.detected ? "text-red-500 animate-pulse" : "text-slate-500"}
                    />
                  </div>
                  <p className="mt-2 text-sm">PM10: <span className="font-bold">{node.pm10.toFixed(1)}</span></p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Logs */}
        <div className="bg-black rounded p-4 overflow-y-auto font-mono text-sm">
          {data.recent_logs.map((log, i) => (
            <div
              key={i}
              className={log.type === "ACTION" ? "text-cyan-400" : log.type === "ALERT" ? "text-red-400" : "text-slate-300"}
            >
              [{log.time}] {log.type}: {log.message}
            </div>
          ))}
        </div>
      </div>

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center">
          <div className="bg-slate-900 p-6 rounded w-80">
            <h3 className="mb-4">Add Sensor</h3>
            <input
              placeholder="Node ID"
              className="w-full mb-2 p-2 bg-slate-800"
              value={newNode.id}
              onChange={e => setNewNode({ ...newNode, id: e.target.value })}
            />
            <input
              placeholder="Location"
              className="w-full mb-4 p-2 bg-slate-800"
              value={newNode.location}
              onChange={e => setNewNode({ ...newNode, location: e.target.value })}
            />
            <button onClick={addSensor} className="bg-cyan-600 px-4 py-2 rounded">Save</button>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ title, value }) {
  return (
    <div className="bg-slate-900 p-4 rounded">
      <p className="text-xs text-slate-400">{title}</p>
      <p className="text-xl font-bold">{value}</p>
    </div>
  );
}
import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, ResponsiveContainer } from "recharts";
import { Camera, Plus, Download } from "lucide-react";

const STATUS_COLORS = {
  good: "#00E400",
  moderate: "#FFFF00",
  unhealthy: "#FF0000",
  hazardous: "#7E0023",
};

const initialData = {
  system_status: "ONLINE",
  timestamp: new Date().toISOString(),
  nodes: [
    {
      id: "Node_1",
      location: "Site A - North Gate",
      pm25: 45.2,
      pm10: 180.5,
      predicted_pm10: 265.2,
      camera_detection: { detected: true, severity: 0.8, confidence: 0.95 },
      status: "unhealthy",
      mitigation_active: true,
    },
  ],
  recent_logs: [
    { time: "10:29:45", type: "ALERT", message: "High PM10 forecast (265 µg/m³) at Node 1" },
    { time: "10:29:50", type: "ACTION", message: "Triggered Sprinklers (Intensity: 90%)" },
  ],
};

function getStatus(pm10) {
  if (pm10 < 50) return "good";
  if (pm10 < 150) return "moderate";
  if (pm10 < 250) return "unhealthy";
  return "hazardous";
}

export default function App() {
  const [data, setData] = useState(initialData);
  const [chartData, setChartData] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [newNode, setNewNode] = useState({ id: "", location: "" });

  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => {
        const updatedNodes = prev.nodes.map(n => {
          const delta = (Math.random() - 0.5) * 20;
          const pm10 = Math.max(10, n.pm10 + delta);
          return {
            ...n,
            pm10,
            predicted_pm10: pm10 + 50 + Math.random() * 30,
            status: getStatus(pm10),
          };
        });

        const timeLabel = new Date().toLocaleTimeString();
        setChartData(cd => [
          ...cd.slice(-29),
          {
            time: timeLabel,
            pm10: updatedNodes[0].pm10,
            forecast: updatedNodes[0].predicted_pm10,
          },
        ]);

        return {
          ...prev,
          timestamp: new Date().toISOString(),
          nodes: updatedNodes,
        };
      });
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const addSensor = () => {
    setData(d => ({
      ...d,
      nodes: [
        ...d.nodes,
        {
          id: newNode.id,
          location: newNode.location,
          pm25: 20,
          pm10: 40,
          predicted_pm10: 70,
          camera_detection: { detected: false },
          status: "good",
        },
      ],
    }));
    setShowModal(false);
    setNewNode({ id: "", location: "" });
  };

  const exportCSV = () => {
    const headers = "time,pm10,forecast\n";
    const rows = chartData.map(r => `${r.time},${r.pm10.toFixed(2)},${r.forecast.toFixed(2)}`).join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "dustvision_pm10_history.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-screen grid grid-cols-[260px_1fr] bg-slate-950 text-slate-100">
      {/* Sidebar */}
      <aside className="bg-slate-900 p-6 flex flex-col gap-6">
        <h1 className="text-xl font-bold">DustVision AI</h1>
        <nav className="flex flex-col gap-3 text-slate-300">
          <span className="font-medium text-white">Dashboard</span>
          <span>Sensor Network</span>
          <span>Compliance Reports</span>
          <span>Settings</span>
        </nav>
        <button
          onClick={() => setShowModal(true)}
          className="mt-auto flex items-center gap-2 bg-cyan-600 hover:bg-cyan-700 px-4 py-2 rounded"
        >
          <Plus size={16} /> Add Sensor
        </button>
      </aside>

      {/* Dashboard Grid */}
      <div className="grid grid-rows-[auto_1fr_220px] gap-4 p-6 overflow-hidden">
        {/* Header */}
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-semibold">City Air Quality Dashboard</h2>
          <button
            onClick={exportCSV}
            className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded"
          >
            <Download size={16} /> Export CSV
          </button>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-[1fr_420px] gap-4 overflow-hidden">
          {/* Left Column */}
          <div className="grid grid-rows-[auto_1fr] gap-4">
            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
              <Stat title="System Health" value={data.system_status} />
              <Stat title="Active Sensors" value={data.nodes.length} />
              <Stat title="Avg PM10" value={Math.round(data.nodes.reduce((a, b) => a + b.pm10, 0) / data.nodes.length)} />
              <Stat title="Active Alerts" value={data.recent_logs.filter(l => l.type === "ALERT").length} />
            </div>

            {/* Chart */}
            <div className="bg-slate-900 rounded p-4">
              <h3 className="mb-2">PM10 – Real Time vs AI Forecast</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <ReferenceLine y={250} stroke="#FF0000" strokeDasharray="5 5" />
                  <Line type="monotone" dataKey="pm10" strokeWidth={2} />
                  <Line type="monotone" dataKey="forecast" strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Sensor Column */}
          <div className="bg-slate-900 rounded p-4 overflow-y-auto">
            <h3 className="mb-3">Live Sensor Feed</h3>
            <div className="flex flex-col gap-3">
              {data.nodes.map(node => (
                <div
                  key={node.id}
                  className="p-3 rounded border"
                  style={{ borderColor: STATUS_COLORS[node.status] }}
                >
                  <div className="flex justify-between items-center">
                    <div>
                      <p className="font-semibold">{node.id}</p>
                      <p className="text-xs text-slate-400">{node.location}</p>
                    </div>
                    <Camera
                      className={node.camera_detection.detected ? "text-red-500 animate-pulse" : "text-slate-500"}
                    />
                  </div>
                  <p className="mt-2 text-sm">PM10: <span className="font-bold">{node.pm10.toFixed(1)}</span></p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Logs */}
        <div className="bg-black rounded p-4 overflow-y-auto font-mono text-sm">
          {data.recent_logs.map((log, i) => (
            <div
              key={i}
              className={log.type === "ACTION" ? "text-cyan-400" : log.type === "ALERT" ? "text-red-400" : "text-slate-300"}
            >
              [{log.time}] {log.type}: {log.message}
            </div>
          ))}
        </div>
      </div>

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center">
          <div className="bg-slate-900 p-6 rounded w-80">
            <h3 className="mb-4">Add Sensor</h3>
            <input
              placeholder="Node ID"
              className="w-full mb-2 p-2 bg-slate-800"
              value={newNode.id}
              onChange={e => setNewNode({ ...newNode, id: e.target.value })}
            />
            <input
              placeholder="Location"
              className="w-full mb-4 p-2 bg-slate-800"
              value={newNode.location}
              onChange={e => setNewNode({ ...newNode, location: e.target.value })}
            />
            <button onClick={addSensor} className="bg-cyan-600 px-4 py-2 rounded">Save</button>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ title, value }) {
  return (
    <div className="bg-slate-900 p-4 rounded">
      <p className="text-xs text-slate-400">{title}</p>
      <p className="text-xl font-bold">{value}</p>
    </div>
  );
}
