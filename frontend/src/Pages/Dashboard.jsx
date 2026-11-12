import { useNavigate } from 'react-router-dom'

function Dashboard() {
  const navigate = useNavigate()

  const handleLogout = () => {
    navigate('/login')
  }

  return (
    <div className='dashboard-page'>

      <div className='dashboard-header'>

        <div>
          <h1>Dashboard</h1>
          <p>Welcome to the Medical Image Analysis System.</p>
        </div>

        <button
          type='button'
          onClick={handleLogout}
          className='logout-button'
        >
          Logout
        </button>

      </div>

      <div className='dashboard-cards'>

        <div
          className='dashboard-card'
          onClick={() => navigate('/cases')}
        >
          <h2>Cases</h2>
          <p>Manage medical image cases.</p>
          <strong>View Cases →</strong>
        </div>

        <div className='dashboard-card'>
          <h2>Image Processing</h2>
          <p>Upload and process medical images.</p>
          <strong>Available through Cases</strong>
        </div>

      </div>

    </div>
  )
}

export default Dashboard