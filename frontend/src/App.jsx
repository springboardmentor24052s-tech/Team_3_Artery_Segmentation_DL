import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Dashboard from './Pages/Dashboard'
import Cases from './Pages/Cases'
import Login from './Pages/Login'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<Navigate to='/login' />} />
        <Route path='/login' element={<Login />} />
        <Route path='/dashboard' element={<Dashboard />} />
        <Route path='/cases' element={<Cases />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App




/*function App() {
  return (
    <div className="dashboard">
      <header className="topbar">
        <div>
          <h1>Artery Segmentation</h1>
          <p>Medical Image Analysis System</p>
        </div>

        <button className="logout-button">
          Logout
        </button>
      </header>

      <main className="dashboard-content">
        <h2>Dashboard</h2>

        <p className="welcome-text">
          Welcome to the medical image analysis system.
        </p>

        <div className="dashboard-cards">
          <div className="dashboard-card">
            <h3>Cases</h3>
            <p>Manage your medical image cases.</p>

            <button>
              View Cases
            </button>
          </div>

          <div className="dashboard-card">
            <h3>Image Upload</h3>
            <p>
              Upload a medical image for processing.
            </p>

            <button>
              Upload Image
            </button>
          </div>

          <div className="dashboard-card">
            <h3>Analysis Results</h3>
            <p>
              Analysis results will appear here after processing.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;*/

/*import Cases from "./Pages/Cases";

function App() {
  return <Cases />;
}

export default App;*/

/*import CaseDetails from './Pages/CaseDetails'

function App() {
  return <CaseDetails />
}

export default App*/

/*function App() {
  const handleClick = () => {
    alert("React button is working!");
  };

  return (
    <div style={{ padding: "40px" }}>
      <h1>Button Test</h1>

      <button onClick={handleClick}>
        TEST BUTTON
      </button>
    </div>
  );
}

export default App;*/
