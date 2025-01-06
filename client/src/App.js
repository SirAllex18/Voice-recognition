import { BrowserRouter, Navigate, Routes, Route } from "react-router-dom";
import HomePage from "./scenes/index"

function App() {
  return (
    <div className="app">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
        </Routes>
      
      </BrowserRouter>

    </div>
  );
}

export default App;
