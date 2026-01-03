import React, { useState, useEffect } from 'react';
import { Bell, ShieldCheck, Moon, Thermometer, AlertTriangle, Phone, Settings, Activity, Eye, Play, Square, RotateCcw } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const App = () => {
  // 앱 상태 관리
  const [isEmergency, setIsEmergency] = useState(false);
  const [emergencyData, setEmergencyData] = useState(null);
  const [sleepAnalysisData, setSleepAnalysisData] = useState(null);
  const [showSleepAnalysis, setShowSleepAnalysis] = useState(false);
  const [isSleepMode, setIsSleepMode] = useState(false); // 수면 모드 상태
  const [fallDetectionData, setFallDetectionData] = useState({
    is_running: false,
    fall_count: 0,
    last_fall_time: 0,
    current_fps: 0,
    status: 'stopped',
    last_fall_image: null  // 낙상 이미지 경로
  });
  
  const [backendConnected, setBackendConnected] = useState(true); // 백엔드 연결 상태

  const [logs, setLogs] = useState([
    { id: 1, time: "오전 08:00", message: "Monoculus 시스템 모니터링 시작" },
    { id: 2, time: "오전 02:45", message: "뒤척임 감지: IoT 온도 최적화 실행" }
  ]);

  // 백엔드에서 fall detection 데이터 가져오기
  const fetchFallDetectionData = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5초 타임아웃
      
      const response = await fetch('http://localhost:5000/api/status', {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error('Backend not responding');
      }
      
      const data = await response.json();
      setFallDetectionData(data);
      setBackendConnected(true); // 연결 성공

      // 낙상이 감지되면 위급 상황으로 전환
      if (data.fall_count > 0 && data.last_fall_time > fallDetectionData.last_fall_time) {
        setIsEmergency(true);
        setEmergencyData({
          type: "낙상 감지",
          time: new Date(data.last_fall_time * 1000).toLocaleTimeString(),
          reason: `NPU가 급격한 위치 변화를 감지했습니다`,
          imgUrl: data.last_fall_image ? `http://localhost:5000${data.last_fall_image}` : "https://images.unsplash.com/photo-1516733725897-1aa73b87c8e8?w=800&auto=format&fit=crop",
          isMasked: true
        });
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Backend request timeout');
      } else {
        console.log('Backend connection issue:', error.message);
      }
      setBackendConnected(false); // 연결 실패
    }
  };

  // 주기적으로 데이터 업데이트
  useEffect(() => {
    fetchFallDetectionData(); // 초기 로드
    const interval = setInterval(fetchFallDetectionData, 3000); // 3초마다
    return () => clearInterval(interval);
  }, [isEmergency]);

  // 수면 분석 데이터 가져오기
  const fetchSleepAnalysis = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/sleep-analysis');
      if (response.ok) {
        const data = await response.json();
        setSleepAnalysisData(data);
      }
    } catch (error) {
      console.error('Failed to fetch sleep analysis:', error);
    }
  };

  // 수면 모니터링 시작
  const startSleepMonitoring = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/sleep/start', { method: 'POST' });
      if (response.ok) {
        alert('💤 수면 모니터링이 시작되었습니다.');
        // 주기적으로 수면 데이터 업데이트
        const interval = setInterval(fetchSleepAnalysis, 5000);
        // cleanup을 위해 interval ID 저장
        window.sleepInterval = interval;
      }
    } catch (error) {
      console.error('Failed to start sleep monitoring:', error);
    }
  };

  // 수면 모니터링 중지
  const stopSleepMonitoring = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/sleep/stop', { method: 'POST' });
      if (response.ok) {
        alert('🛑 수면 모니터링이 중지되었습니다.');
        if (window.sleepInterval) {
          clearInterval(window.sleepInterval);
        }
      }
    } catch (error) {
      console.error('Failed to stop sleep monitoring:', error);
    }
  };

  // 위급 상황 시뮬레이션 (테스트용)
  const triggerEmergency = async (type) => {
    if (type === 'fire') {
      // 화재 감지 시뮬레이션
      const audio = new Audio('/emergency-alarm.mp3'); // 경고음 재생
      audio.play().catch(() => {
        // 브라우저가 자동 재생을 차단하는 경우
        alert('🚨 화재 감지! 경고음이 재생됩니다.');
      });
      
      setIsEmergency(true);
      setEmergencyData({
        type: "🔥 화재 감지",
        time: new Date().toLocaleTimeString(),
        reason: "연기 및 높은 온도 감지 (45°C)",
        imgUrl: "https://images.unsplash.com/photo-1583537904458-95965c37e4a5?w=800&auto=format&fit=crop",
        isMasked: false,
        showCall119: true
      });
      return;
    }
    
    if (type === 'sleeping') {
      // 수면 모드 활성화
      setIsSleepMode(true);
      setShowSleepAnalysis(true);
      await startSleepMonitoring();
      
      // 수면 분석 데이터 주기적으로 가져오기
      if (window.sleepInterval) {
        clearInterval(window.sleepInterval);
      }
      window.sleepInterval = setInterval(fetchSleepAnalysis, 5000);
      return;
    }
    
    setIsEmergency(true);
    const mockEvents = {
      fall: {
        type: "낙상 감지",
        time: new Date().toLocaleTimeString(),
        reason: "급격한 위치 변화 및 바닥 충격 감지",
        imgUrl: "https://images.unsplash.com/photo-1516733725897-1aa73b87c8e8?w=800&auto=format&fit=crop",
        isMasked: true,
        showCall119: false
      },
      intruder: {
        type: "침입자 탐지",
        time: new Date().toLocaleTimeString(),
        reason: "미등록 인원 현관 진입 시도",
        imgUrl: "/images/intruder.svg",
        isMasked: false,
        showCall119: false
      }
    };
    setEmergencyData(mockEvents[type]);
  };

  // 119 신고 함수
  const call119 = () => {
    const confirmed = confirm('119에 신고하시겠습니까?\n\n화재 발생 위치와 상황이 자동으로 전달됩니다.');
    if (confirmed) {
      alert('📞 119 신고가 접수되었습니다.\n\n위치: 서울시 강남구\n상황: 화재 감지\n시간: ' + new Date().toLocaleString());
      // 실제로는 여기서 119 API 호출
    }
  };

  // Fall detection 제어 함수들
  const startFallDetection = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch('http://localhost:5000/api/start', { 
        method: 'POST',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        setTimeout(fetchFallDetectionData, 500);
        setBackendConnected(true);
      }
    } catch (error) {
      setBackendConnected(false);
      alert('❌ 백엔드 서버에 연결할 수 없습니다.\n\n해결 방법:\n1. 터미널에서 "bash /root/run_monoculus.sh" 실행\n2. 또는 "cd /root/backend && python app.py" 실행');
    }
  };

  const stopFallDetection = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch('http://localhost:5000/api/stop', { 
        method: 'POST',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        setTimeout(fetchFallDetectionData, 500);
      }
    } catch (error) {
      alert('❌ 백엔드 서버에 연결할 수 없습니다.');
    }
  };

  const resetFallDetection = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch('http://localhost:5000/api/reset', { 
        method: 'POST',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        setTimeout(fetchFallDetectionData, 500);
      }
    } catch (error) {
      alert('❌ 백엔드 서버에 연결할 수 없습니다.');
    }
  };

  const resetStatus = () => {
    setIsEmergency(false);
    setEmergencyData(null);
  };

  return (
    <div className={`min-h-screen font-sans transition-colors duration-700 ${isEmergency ? 'bg-red-50' : 'bg-slate-50'}`}>
      {/* 상단 네비게이션 */}
      <nav className="bg-white/80 backdrop-blur-md sticky top-0 z-10 border-b px-6 py-4 flex justify-between items-center">
        {/* 로고 클릭 시 resetStatus 실행하여 홈으로 이동 */}
        <div
          className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
          onClick={resetStatus}
        >
          <div className="bg-blue-600 p-1.5 rounded-xl">
            <Eye className="text-white" size={20} />
          </div>
          <h1 className="text-xl font-black text-slate-900 tracking-tighter uppercase">MONOCULUS</h1>
          {/* 백엔드 연결 상태 표시 */}
          <div className={`ml-2 w-2 h-2 rounded-full ${backendConnected ? 'bg-green-500' : 'bg-red-500'}`} title={backendConnected ? 'Backend Connected' : 'Backend Disconnected'}></div>
        </div>
        <div className="flex gap-2">
          <button onClick={() => triggerEmergency('fire')} className="text-[9px] font-bold border border-slate-200 px-2 py-1 rounded-lg hover:bg-slate-100">Fire</button>
          <button onClick={() => triggerEmergency('intruder')} className="text-[9px] font-bold border border-slate-200 px-2 py-1 rounded-lg hover:bg-slate-100">Intruder</button>
          <button onClick={() => triggerEmergency('fall')} className="text-[9px] font-bold border border-slate-200 px-2 py-1 rounded-lg hover:bg-slate-100">Falling</button>
          <button onClick={() => triggerEmergency('sleeping')} className="text-[9px] font-bold border border-slate-200 px-2 py-1 rounded-lg hover:bg-slate-100">Sleeping</button>
        </div>
      </nav>

      <main className="max-w-md mx-auto p-6 space-y-6 pb-20">
        {/* 수면 분석 화면 - 수면 모드일 때만 표시 */}
        {isSleepMode && showSleepAnalysis && sleepAnalysisData ? (
          <div className="space-y-6">
            <div className="bg-white rounded-[2rem] p-8 shadow-sm border border-slate-100">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-extrabold text-slate-900 flex items-center gap-2">
                  <Moon className="text-blue-600" size={28} />
                  수면 분석 중...
                </h2>
                <button 
                  onClick={() => {
                    setIsSleepMode(false);
                    setShowSleepAnalysis(false);
                    stopSleepMonitoring();
                  }}
                  className="px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg text-sm font-medium"
                >
                  수면 모니터링 중지
                </button>
              </div>

              {/* 뒤척임 횟수 그래프 */}
              <div className="mb-8">
                <h3 className="text-lg font-bold text-slate-800 mb-4">뒤척임 횟수</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={sleepAnalysisData.toss_turn_history}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{fontSize: 12}} />
                    <YAxis tick={{fontSize: 12}} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="count" stroke="#8b5cf6" strokeWidth={2} name="뒤척임 (회)" />
                  </LineChart>
                </ResponsiveContainer>
                
                {/* 총 뒤척임 횟수 */}
                <div className="mt-4 bg-purple-50 rounded-lg p-4">
                  <p className="text-sm text-slate-600">총 뒤척임 횟수</p>
                  <p className="text-3xl font-bold text-purple-600">
                    {sleepAnalysisData.toss_turn_history.reduce((sum, item) => sum + item.count, 0)}회
                  </p>
                </div>
              </div>

              {/* 온도 변화 그래프 */}
              <div>
                <h3 className="text-lg font-bold text-slate-800 mb-4">실내 온도 변화</h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={sleepAnalysisData.temperature_history}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{fontSize: 12}} />
                    <YAxis domain={[20, 25]} tick={{fontSize: 12}} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="temp" stroke="#ef4444" strokeWidth={2} name="온도 (°C)" />
                  </LineChart>
                </ResponsiveContainer>
                
                {/* 현재 온도 */}
                <div className="mt-4 bg-red-50 rounded-lg p-4">
                  <p className="text-sm text-slate-600">현재 온도</p>
                  <p className="text-3xl font-bold text-red-600">
                    {sleepAnalysisData.temperature_history[sleepAnalysisData.temperature_history.length - 1]?.temp}°C
                  </p>
                </div>
              </div>

              {/* 수면 분석 인사이트 */}
              <div className="mt-6 bg-blue-50 rounded-lg p-4">
                <p className="text-sm font-semibold text-blue-900 mb-2">💡 수면 분석 인사이트</p>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• 오늘 밤 뒤척임이 많았습니다. 침실 온도를 낮춰보세요.</li>
                  <li>• 새벽 3시경 온도가 높아져 수면에 영향을 줄 수 있습니다.</li>
                  <li>• 평균 온도: {(sleepAnalysisData.temperature_history.reduce((sum, item) => sum + item.temp, 0) / sleepAnalysisData.temperature_history.length).toFixed(1)}°C</li>
                </ul>
              </div>
            </div>
          </div>
        ) : !isEmergency && !isSleepMode ? (
          <div className="bg-white rounded-[2rem] p-8 shadow-sm border border-slate-100 text-center space-y-4">
            <div className="relative w-20 h-20 mx-auto">
              <div className={`absolute inset-0 rounded-full animate-ping ${fallDetectionData.is_running ? 'bg-green-400/20' : 'bg-gray-400/20'}`}></div>
              <div className={`relative w-20 h-20 rounded-full flex items-center justify-center shadow-lg ${fallDetectionData.is_running ? 'bg-green-500 shadow-green-200' : 'bg-gray-500 shadow-gray-200'}`}>
                <ShieldCheck className="text-white" size={36} />
              </div>
            </div>
            <div>
              <h2 className="text-2xl font-extrabold text-slate-900">
                현재 상태: {fallDetectionData.is_running ? '감지 중' : '중지됨'}
              </h2>
              <p className="text-slate-500 text-sm mt-1">
                {fallDetectionData.is_running ? 'NPU가 실시간으로 낙상을 감지하고 있습니다.' : '시스템이 대기 중입니다.'}
              </p>
            </div>
          </div>
        ) : (
          <div className="bg-red-600 rounded-[2rem] p-8 shadow-2xl text-center space-y-4 animate-pulse">
            <div className="w-20 h-20 bg-white/20 rounded-full flex items-center justify-center mx-auto">
              <AlertTriangle className="text-white" size={40} />
            </div>
            <div>
              <h2 className="text-2xl font-extrabold text-white">위급 상황 발생</h2>
              <p className="text-white/80 text-sm mt-1">즉시 현장을 확인해 주세요.</p>
            </div>
          </div>
        )}

        {/* Fall Detection 제어 버튼들 - 수면 모드가 아닐 때만 표시 */}
        {!isSleepMode && (
          <div className="bg-white rounded-[1.5rem] p-5 shadow-sm border border-slate-100">
            <h3 className="font-bold text-slate-700 text-sm mb-4">Fall Detection 제어</h3>
            <div className="grid grid-cols-3 gap-3">
              <button
                onClick={startFallDetection}
                disabled={fallDetectionData.is_running}
                className="bg-green-500 text-white py-3 rounded-lg font-bold flex flex-col items-center justify-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play size={16} />
                <span className="text-xs">시작</span>
              </button>
              <button
                onClick={stopFallDetection}
                disabled={!fallDetectionData.is_running}
                className="bg-red-500 text-white py-3 rounded-lg font-bold flex flex-col items-center justify-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Square size={16} />
                <span className="text-xs">중지</span>
              </button>
              <button
                onClick={resetFallDetection}
                className="bg-blue-500 text-white py-3 rounded-lg font-bold flex flex-col items-center justify-center gap-1"
              >
                <RotateCcw size={16} />
                <span className="text-xs">리셋</span>
              </button>
            </div>
            <div className="mt-4 text-center text-sm text-slate-600">
              상태: {fallDetectionData.status} | 낙상: {fallDetectionData.fall_count}회 | FPS: {fallDetectionData.current_fps.toFixed(1)}
            </div>
            
            {/* 디버그 정보 */}
            {fallDetectionData.is_running && fallDetectionData.debug_info && (
            <div className="mt-4 p-3 bg-slate-50 rounded-lg text-xs space-y-1">
              <div className="font-bold text-slate-700 mb-2">🔍 실시간 디버그 정보</div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <span className="text-slate-500">감지된 사람:</span>
                  <span className="ml-2 font-bold text-slate-900">{fallDetectionData.debug_info.people_detected}명</span>
                </div>
                <div>
                  <span className="text-slate-500">머리 Y:</span>
                  <span className="ml-2 font-bold text-slate-900">{fallDetectionData.debug_info.head_y_position.toFixed(0)}px</span>
                </div>
                <div>
                  <span className="text-slate-500">낙하 속도:</span>
                  <span className={`ml-2 font-bold ${Math.abs(fallDetectionData.debug_info.head_velocity) > 1200 ? 'text-red-600' : 'text-slate-900'}`}>
                    {fallDetectionData.debug_info.head_velocity.toFixed(0)} px/s
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">낙하 거리:</span>
                  <span className={`ml-2 font-bold ${fallDetectionData.debug_info.vertical_distance > 180 ? 'text-red-600' : 'text-slate-900'}`}>
                    {fallDetectionData.debug_info.vertical_distance.toFixed(0)}px
                  </span>
                </div>
              </div>
              <div className="mt-2 text-[10px] text-slate-400">
                임계값: 속도 &gt; 1200 px/s, 거리 &gt; 180px
              </div>
            </div>
            )}
          </div>
        )}

        {/* 2. 메인 컨텐츠 영역 - 수면 모드가 아닐 때만 표시 */}
        {!isEmergency && !isSleepMode ? (
          <>
            <div className="grid grid-cols-2 gap-4">
              {/* 수면 리포트 */}
              <div className="bg-white rounded-[1.5rem] p-5 shadow-sm border border-slate-100">
                <div className="flex items-center gap-2 mb-3">
                  <div className="p-1 bg-indigo-50 rounded-lg">
                    <Moon className="text-indigo-500" size={16} />
                  </div>
                  <h3 className="font-bold text-slate-700 text-sm">수면 분석</h3>
                </div>
                <div className="space-y-1">
                  <p className="text-3xl font-black text-slate-900">0회</p>
                  <p className="text-xs text-indigo-500 font-bold tracking-tight">뒤척임</p>
                </div>
              </div>

              {/* IoT 온도 */}
              <div className="bg-white rounded-[1.5rem] p-5 shadow-sm border border-slate-100">
                <div className="flex items-center gap-2 mb-3">
                  <div className="p-1 bg-orange-50 rounded-lg">
                    <Thermometer className="text-orange-500" size={16} />
                  </div>
                  <h3 className="font-bold text-slate-700 text-sm">시스템 상태</h3>
                </div>
                <div className="space-y-1">
                  <p className="text-xl font-black text-slate-900">{fallDetectionData.current_fps.toFixed(1)} FPS</p>
                  <p className="text-xs text-orange-500 font-bold tracking-tight">실시간 처리</p>
                </div>
              </div>
            </div>

            {/* 실시간 타임라인 */}
            <div className="bg-white rounded-[1.5rem] p-6 shadow-sm border border-slate-100">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Activity className="text-blue-500" size={18} />
                  <h3 className="font-bold text-slate-800 tracking-tight">Monoculus 분석 로그</h3>
                </div>
              </div>
              <div className="space-y-4">
                {logs.map(log => (
                  <div key={log.id} className="flex gap-4 items-start group">
                    <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-1.5 shadow-[0_0_8px_rgba(59,130,246,0.8)] group-hover:scale-125 transition-transform"></div>
                    <div className="flex-1">
                      <p className="text-sm text-slate-700 leading-tight font-medium">{log.message}</p>
                      <p className="text-[10px] text-slate-400 mt-1 uppercase font-mono">{log.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : (
          /* 위급 상황 캡처 뷰 */
          <div className="bg-white rounded-[2rem] overflow-hidden shadow-2xl border-4 border-red-500/20">
            <div className="relative aspect-video bg-black">
              <img
                src={emergencyData.imgUrl}
                alt="NPU Captured"
                className={`w-full h-full object-cover transition-all duration-1000 ${emergencyData.isMasked ? 'blur-2xl opacity-50' : 'opacity-90'}`}
              />
              <div className="absolute top-4 left-4 flex gap-2">
                <div className="bg-red-600 text-white text-[10px] font-bold px-2 py-1 rounded flex items-center gap-1">
                  <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse"></div> LIVE FEED
                </div>
                {emergencyData.isMasked && (
                  <div className="bg-blue-600 text-white text-[10px] font-bold px-2 py-1 rounded">PRIVACY ON</div>
                )}
              </div>
              {emergencyData.isMasked && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <p className="text-white/60 text-[10px] font-medium tracking-widest text-center px-10">
                    NPU 온디바이스 기술로 사용자의 프라이버시를 보호 중입니다.
                  </p>
                </div>
              )}
            </div>
            <div className="p-6">
              <div className="mb-6">
                <div className="flex justify-between items-end mb-2">
                  <h3 className="text-3xl font-black text-red-600 tracking-tighter">{emergencyData.type}</h3>
                  <span className="text-slate-400 text-xs font-mono">{emergencyData.time}</span>
                </div>
                <div className="bg-red-50 p-4 rounded-2xl border border-red-100">
                  <p className="text-red-900 font-bold text-sm leading-relaxed">{emergencyData.reason}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <button 
                  onClick={() => {
                    const audio = new Audio('/emergency-alarm.mp3');
                    audio.play().catch(() => alert('🚨 경고음이 재생됩니다!'));
                  }}
                  className="bg-slate-900 text-white py-4 rounded-2xl font-bold flex flex-col items-center justify-center gap-1 active:scale-95 transition-transform"
                >
                  <Bell size={20} />
                  <span className="text-xs">경고음</span>
                </button>
                {emergencyData.showCall119 ? (
                  <button 
                    onClick={call119}
                    className="bg-red-600 text-white py-4 rounded-2xl font-bold flex flex-col items-center justify-center gap-1 shadow-lg shadow-red-200 active:scale-95 transition-transform animate-pulse"
                  >
                    <Phone size={20} />
                    <span className="text-xs">119 신고</span>
                  </button>
                ) : (
                  <button className="bg-red-600 text-white py-4 rounded-2xl font-bold flex flex-col items-center justify-center gap-1 shadow-lg shadow-red-200 active:scale-95 transition-transform">
                    <Phone size={20} />
                    <span className="text-xs">보호자 연락</span>
                  </button>
                )}
              </div>

              <button
                onClick={resetStatus}
                className="w-full mt-6 py-2 text-slate-400 text-xs font-bold uppercase tracking-widest hover:text-slate-600 transition-colors"
              >
                상황 해제 및 시스템 복구
              </button>
            </div>
          </div>
        )}
      </main>

      {/* 바닥 안내 문구 */}
      <footer className="fixed bottom-0 left-0 right-0 p-4 bg-white/50 backdrop-blur-sm border-t border-slate-100">
        <div className="max-w-md mx-auto text-center">
          <p className="text-[10px] text-slate-400 font-medium">
            Powered by Monoculus NPU Engine • 100% On-Device Encryption
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
