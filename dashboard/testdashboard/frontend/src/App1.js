import React, { useState, useEffect, useMemo } from "react";
import {
  Container,
  Typography,
  Grid,
  Paper,
  CircularProgress,
  useTheme,
  useMediaQuery,
  Chip,
  Stack,
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
} from "recharts";
import {
  Security,
  Lock,
  NotificationsActive,
  Timeline,
  PieChart as PieChartIcon,
} from "@mui/icons-material";
import { styled } from "@mui/material/styles";

// Professional Color Palette
const colors = {
  primary: "#4F46E5", // Indigo-600
  secondary: "#06B6D4", // Cyan-500
  success: "#10B981", // Emerald-500
  warning: "#F59E0B", // Amber-500
  danger: "#EF4444", // Red-500
  background: "#0F172A", // Slate-900
  surface: "#1E293B", // Slate-800
  surface2: "#334155", // Slate-700
  textPrimary: "#F8FAFC", // Slate-50
  textSecondary: "#94A3B8", // Slate-400
};

const ProfessionalPaper = styled(Paper)(({ theme }) => ({
  backgroundColor: colors.surface,
  color: colors.textPrimary,
  padding: theme.spacing(3),
  borderRadius: "16px",
  boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.25)",
  transition: "transform 0.2s ease, box-shadow 0.2s ease",
  "&:hover": {
    transform: "translateY(-2px)",
    boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.3)",
  },
  [theme.breakpoints.down("sm")]: {
    padding: theme.spacing(2),
  },
}));

const Dashboard = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isMobileDrawerOpen, setIsMobileDrawerOpen] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  const toggleSidebar = () => {
    if (isMobile) {
      setIsMobileDrawerOpen(!isMobileDrawerOpen);
    } else {
      setIsSidebarOpen(!isSidebarOpen);
    }
  };

  // Close mobile drawer when resizing to desktop
  useEffect(() => {
    if (!isMobile && isMobileDrawerOpen) {
      setIsMobileDrawerOpen(false);
    }
  }, [isMobile, isMobileDrawerOpen]);

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

    const interval = setInterval(fetchData, 5000);

    // Cleanup interval on component unmount
    return () => clearInterval(interval);
  }, []);

  const summaryData = useMemo(
    () => [
      {
        title: "Total Threats",
        // Filter out entries with "None" severity
        value: data.filter((d) => d.severity !== "None").length,
        icon: <Security fontSize="large" />,
        trend: "Updated Live",
      },
      {
        title: "Critical Alerts",
        value: data.filter(
          (d) =>
            d.severity !== "None" &&
            (d.severity === "Critical" || d.severity === "High")
        ).length,
        icon: <NotificationsActive fontSize="large" />,
        trend: "Real-time Monitoring",
      },
      {
        title: "Number of Indexed Onion Sites",
        // Count all entries regardless of severity
        value: data.length,
        icon: <Lock fontSize="large" />,
        trend: "Constantly Updated",
      },
    ],
    [data]
  ); // Recalculate summaryData when data changes

  const severityDistribution = [
    {
      severity: "Critical",
      value: data.filter((d) => d.severity === "Critical").length,
    },
    {
      severity: "High",
      value: data.filter((d) => d.severity === "High").length,
    },
    {
      severity: "Medium",
      value: data.filter((d) => d.severity === "Medium").length,
    },
    { severity: "Low", value: data.filter((d) => d.severity === "Low").length },
    {
      severity: "None",
      value: data.filter((d) => d.severity === "None").length,
    },
  ];

  const processedLineData = useMemo(() => {
    let critical = 0;
    let high = 0;
    let medium = 0;
    let low = 0;
    let none = 0;

    return data.map((entry) => {
      switch (entry.severity) {
        case "Critical":
          critical += 1;
          break;
        case "High":
          high += 1;
          break;
        case "Medium":
          medium += 1;
          break;
        case "Low":
          low += 1;
          break;
        case "None":
          none += 1;
          break;
        default:
          break;
      }

      return {
        id: entry.id,
        Critical: critical,
        High: high,
        Medium: medium,
        Low: low,
        None: none,
        indexedOnionSites: entry.indexedOnionSites,
      };
    });
  }, [data]);

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
    <Box sx={{ display: "flex" }}>
      {/* Sidebar Toggle Button */}
      <IconButton
        color="inherit"
        onClick={toggleSidebar}
        sx={{
          position: "fixed",
          top: 16,
          left: 16,
          zIndex: theme.zIndex.drawer + 1,
          color: colors.textPrimary,
          backgroundColor: colors.surface2,
          "&:hover": {
            backgroundColor: colors.surface2,
          },
          display: "flex",
        }}
      >
        <MenuIcon />
      </IconButton>

      {/* Sidebar Navigation */}
      <Paper
        component="nav"
        sx={{
          width: 240,
          minHeight: "100vh",
          backgroundColor: colors.surface,
          borderRight: `1px solid ${colors.surface2}`,
          position: "fixed",
          left: 0,
          top: 0,
          zIndex: theme.zIndex.drawer,
          transition: theme.transitions.create("transform", {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          transform: {
            xs: `translateX(${isMobileDrawerOpen ? 0 : "-100%"})`,
            md: `translateX(${isSidebarOpen ? 0 : "-100%"})`,
          },
          [theme.breakpoints.up("md")]: {
            transform: `translateX(${isSidebarOpen ? 0 : "-100%"})`,
          },
        }}
      >
        {/* Close Button for Mobile */}
        <Box
          sx={{
            display: "flex",
            justifyContent: "flex-end",
            p: 1,
            borderBottom: `1px solid ${colors.surface2}`,
          }}
        >
          <IconButton
            onClick={toggleSidebar}
            sx={{ color: colors.textPrimary }}
          >
            <ChevronLeftIcon />
          </IconButton>
        </Box>

        <List sx={{ p: 2 }}>
          <ListItem disablePadding>
            <ListItemButton
              sx={{
                color: colors.textPrimary,
                "&:hover": { backgroundColor: colors.surface2 },
                borderRadius: "8px",
                mb: 1,
              }}
            >
              <ListItemIcon>
                <Timeline sx={{ color: colors.textPrimary }} />
              </ListItemIcon>
              <ListItemText primary="Dashboard" />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton
              sx={{
                color: colors.textPrimary,
                "&:hover": { backgroundColor: colors.surface2 },
                borderRadius: "8px",
              }}
            >
              <ListItemIcon>
                <Lock sx={{ color: colors.textPrimary }} />
              </ListItemIcon>
              <ListItemText primary="Indexed Sites" />
            </ListItemButton>
          </ListItem>
        </List>
      </Paper>

      {/* Overlay for Mobile */}
      {isMobile && isMobileDrawerOpen && (
        <Box
          sx={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0,0,0,0.5)",
            zIndex: theme.zIndex.drawer - 1,
            [theme.breakpoints.up("md")]: { display: "none" },
          }}
          onClick={toggleSidebar}
        />
      )}

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          transition: theme.transitions.create("margin", {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
          marginLeft: {
            xs: 0,
            md: isSidebarOpen ? "240px" : 0,
          },
          width: {
            md: `calc(100% - ${isSidebarOpen ? 240 : 0}px)`,
          },
          position: "relative",
          py: 4,
          backgroundColor: colors.background,
          minHeight: "100vh",
        }}
      >
        <Container maxWidth="xl">
          <Stack spacing={4}>
            {/* Header */}
            <Stack
              spacing={1}
              sx={{ pl: 8 }} // Shift text slightly right
            >
              <Typography
                variant="h4"
                sx={{
                  color: colors.textPrimary,
                  fontWeight: 700,
                  letterSpacing: "-0.5px",
                }}
              >
                DarkEye
              </Typography>
              <Typography
                variant="body1"
                sx={{
                  color: colors.textSecondary,
                  fontSize: "0.9rem",
                }}
              >
                Threat Intelligence Dashboard
              </Typography>
            </Stack>

            {/* Summary Cards */}
            <Grid container spacing={3}>
              {summaryData.map((item, index) => (
                <Grid item xs={12} md={4} key={index}>
                  <ProfessionalPaper>
                    <Stack direction="row" spacing={2} alignItems="center">
                      <div
                        style={{
                          backgroundColor: colors.primary + "20",
                          borderRadius: "12px",
                          padding: "12px",
                          display: "flex",
                          alignItems: "center",
                        }}
                      >
                        {React.cloneElement(item.icon, {
                          sx: {
                            color: colors.primary,
                            fontSize: isMobile ? "1.8rem" : "2rem",
                          },
                        })}
                      </div>
                      <Stack>
                        <Typography
                          variant="subtitle1"
                          sx={{
                            color: colors.textSecondary,
                            fontSize: "0.9rem",
                          }}
                        >
                          {item.title}
                        </Typography>
                        <Typography
                          variant="h3"
                          sx={{
                            color: colors.textPrimary,
                            fontWeight: 700,
                            lineHeight: 1.2,
                          }}
                        >
                          {item.value}
                        </Typography>
                        <Typography
                          variant="caption"
                          sx={{
                            color: colors.success,
                            fontWeight: 500,
                            fontSize: "0.75rem",
                          }}
                        >
                          {item.trend}
                        </Typography>
                      </Stack>
                    </Stack>
                  </ProfessionalPaper>
                </Grid>
              ))}
            </Grid>

            {/* Data Visualization Section */}
            <Grid container spacing={3}>
              {/* Line Chart */}
              <Grid item xs={12} lg={8}>
                <ProfessionalPaper>
                  <Stack spacing={2}>
                    <Typography
                      variant="h6"
                      sx={{
                        color: colors.textPrimary,
                        display: "flex",
                        alignItems: "center",
                        gap: 1,
                      }}
                    >
                      <Timeline sx={{ color: colors.primary }} />
                      Total Threats & Indexed Onion Sites
                    </Typography>
                    <div style={{ height: isMobile ? "250px" : "300px" }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={processedLineData}>
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke={colors.textSecondary}
                          />
                          <XAxis
                            dataKey="id"
                            stroke={colors.textSecondary}
                            tick={{ fontSize: isMobile ? 12 : 14 }}
                          />
                          <YAxis
                            stroke={colors.textSecondary}
                            tick={{ fontSize: isMobile ? 12 : 14 }}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: colors.surface2,
                              border: "none",
                              borderRadius: "8px",
                              boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
                            }}
                          />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="Critical"
                            name="Critical"
                            stroke={colors.danger}
                            strokeWidth={1}
                            dot={{ fill: colors.danger, strokeWidth: 1 }}
                            activeDot={{ r: 6 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="High"
                            name="High"
                            stroke={colors.warning}
                            strokeWidth={1}
                            dot={{ fill: colors.warning, strokeWidth: 1 }}
                            activeDot={{ r: 6 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="Medium"
                            name="Medium"
                            stroke={colors.secondary}
                            strokeWidth={1}
                            dot={{ fill: colors.secondary, strokeWidth: 1 }}
                            activeDot={{ r: 6 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="Low"
                            name="Low"
                            stroke={colors.success}
                            strokeWidth={1}
                            dot={{ fill: colors.success, strokeWidth: 1 }}
                            activeDot={{ r: 6 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="None"
                            name="None"
                            stroke="#6B7280"
                            strokeWidth={2}
                            dot={{ fill: "#6B7280", strokeWidth: 2 }}
                            activeDot={{ r: 6 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="indexedOnionSites"
                            name="Indexed Onion Sites"
                            stroke={colors.primary}
                            strokeWidth={2}
                            dot={{ fill: colors.primary, strokeWidth: 2 }}
                            activeDot={{ r: 6 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </Stack>
                </ProfessionalPaper>
              </Grid>

              {/* Severity Distribution Pie Chart */}
              <Grid item xs={12} lg={4}>
                <ProfessionalPaper>
                  <Stack spacing={2}>
                    <Typography
                      variant="h6"
                      sx={{
                        color: colors.textPrimary,
                        display: "flex",
                        alignItems: "center",
                        gap: 1,
                      }}
                    >
                      <PieChartIcon sx={{ color: colors.primary }} />
                      Severity Distribution
                    </Typography>
                    <div style={{ height: isMobile ? "250px" : "300px" }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={severityDistribution}
                            dataKey="value"
                            nameKey="severity"
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            innerRadius={60}
                            paddingAngle={2}
                          >
                            {severityDistribution.map((entry, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={
                                  entry.severity === "Critical"
                                    ? colors.danger
                                    : entry.severity === "High"
                                    ? colors.warning
                                    : entry.severity === "Medium"
                                    ? colors.secondary
                                    : entry.severity === "Low"
                                    ? colors.success
                                    : "#6B7280" // None
                                }
                              />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{
                              backgroundColor: colors.surface2,
                              border: "none",
                              borderRadius: "8px",
                            }}
                          />
                          <Legend
                            layout="vertical"
                            verticalAlign="middle"
                            align="right"
                            wrapperStyle={{
                              paddingLeft: "20px",
                              color: colors.textPrimary,
                            }}
                            formatter={(value) => (
                              <span style={{ color: colors.textPrimary }}>
                                {value}
                              </span>
                            )}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </Stack>
                </ProfessionalPaper>
              </Grid>

              {/* Recent Alerts */}
              <Grid item xs={12}>
                <ProfessionalPaper>
                  <Typography variant="h6" sx={{ color: colors.textPrimary }}>
                    Security Alerts
                  </Typography>
                  <Box
                    sx={{
                      maxHeight: "400px",
                      overflowY: "auto",
                      paddingRight: "8px",
                    }}
                  >
                    <Stack spacing={2}>
                      {data.map((alert, index) => (
                        <Paper
                          key={index}
                          sx={{
                            p: 2,
                            borderRadius: "12px",
                            backgroundColor: colors.surface2,
                          }}
                        >
                          <Stack
                            direction="row"
                            spacing={2}
                            alignItems="center"
                          >
                            <Chip
                              label={alert.severity.toUpperCase()}
                              sx={{
                                fontWeight: 600,
                                color: colors.textPrimary,
                                backgroundColor:
                                  alert.severity === "Critical"
                                    ? "#FF3B30"
                                    : alert.severity === "High"
                                    ? "#B71C1C"
                                    : alert.severity === "Medium"
                                    ? "#F59E0B"
                                    : alert.severity === "Low"
                                    ? "#10B981"
                                    : alert.severity === "None"
                                    ? "#6B7280"
                                    : "#10B981",
                              }}
                            />
                            <Stack>
                              <Typography
                                variant="body1"
                                sx={{
                                  color: colors.textPrimary,
                                  fontWeight: 500,
                                }}
                              >
                                URL: {alert.onionsite_url}
                              </Typography>
                              <Typography
                                variant="caption"
                                sx={{ color: colors.textSecondary }}
                              >
                                ID: {alert.id} | Severity: {alert.severity}
                              </Typography>
                            </Stack>
                          </Stack>
                        </Paper>
                      ))}
                    </Stack>
                  </Box>
                </ProfessionalPaper>
              </Grid>
            </Grid>
          </Stack>
        </Container>
      </Box>
    </Box>
  );
};

export default Dashboard;