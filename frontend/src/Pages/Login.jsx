import { useNavigate } from 'react-router-dom'

function Login() {
  const navigate = useNavigate()

  const handleLogin = (e) => {
    e.preventDefault()
    navigate('/dashboard')
  }

  return (
    <div className='login-page'>
      <div className='login-card'>

        <h1>Artery Segmentation</h1>

        <p>Medical Image Analysis System</p>

        <form onSubmit={handleLogin}>

          <div className='form-group'>
            <label>Username</label>

            <input
              type='text'
              placeholder='Enter username'
            />
          </div>

          <div className='form-group'>
            <label>Password</label>

            <input
              type='password'
              placeholder='Enter password'
            />
          </div>

          <button type='submit'>
            Login
          </button>

        </form>

      </div>
    </div>
  )
}

export default Login