import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import Register from "./components/Register";
import reportWebVitals from "./reportWebVitals";
import Admin from "./components/Admin";
import User from "./components/User";
import { BrowserRouter, Routes, Route } from "react-router-dom";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<Register />} />
      <Route path="/admin" element={<Admin />} />
      <Route path="/user" element={<User />} />
    </Routes>
  </BrowserRouter>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
