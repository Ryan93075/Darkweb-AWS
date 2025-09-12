import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./Dashboard";
import IndexedSites from "./IndexedSites";
import Layout from "./Layout";
import { Container, CircularProgress } from "@mui/material";
import { colors } from "./colors"; // Ensure colors are exported from this file

const App = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("http://localhost:5000/api/threats");
        const result = await response.json();
        setData(result.reverse()); // Reverse the data to show newest first
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    // Fetch data initially
    fetchData();

    // Polling every 2 seconds to fetch new data
    const interval = setInterval(fetchData, 2000);

    // Cleanup interval on component unmount
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Container
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
          backgroundColor: colors.background,
        }}
      >
        <CircularProgress
          size={80}
          thickness={4}
          sx={{ color: colors.primary }}
        />
      </Container>
    );
  }

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard data={data} loading={loading} />} />
          <Route path="indexed-sites" element={<IndexedSites data={data} />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;