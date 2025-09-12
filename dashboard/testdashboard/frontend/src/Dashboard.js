import React, { useMemo, useState, useEffect } from "react";
import {
  Container,
  Typography,
  Grid,
  CircularProgress,
  Chip,
  Stack,
  Box,
  Paper,
  useTheme,
  useMediaQuery,
  IconButton
} from "@mui/material";
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
import CloseIcon from "@mui/icons-material/Close";
import WarningIcon from "@mui/icons-material/Warning";

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

const Dashboard = ({ data, loading }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  const [newAlerts, setNewAlerts] = useState([]);
  const [openAlert, setOpenAlert] = useState(false);
  const [previousData, setPreviousData] = useState([]);

  useEffect(() => {
    if (data.length > previousData.length) {
      const newEntries = data.slice(previousData.length);
      const highAlerts = newEntries.filter((entry) =>
        ["High"].includes(entry.severity)
      );
  
      if (highAlerts.length > 0) {
        setNewAlerts((prev) => [...prev, ...highAlerts]);
        setOpenAlert(true);
      }
    }
    setPreviousData(data);
  }, [data, previousData.length]); // ✅ Add `previousData.length` to the dependency array
  

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
        title: "High Alerts",
        // Already excludes "None" by only including High
        value: data.filter(
          (d) => d.severity !== "None" && d.severity === "High"
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
    let high = 0;
    let medium = 0;
    let low = 0;
    let none = 0;

    return data.map((entry) => {
      switch (entry.severity) {
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
    <Box
      component="main"
      sx={{
        flexGrow: 1,
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
                          dataKey="High"
                          name="High"
                          stroke={colors.warning}
                          strokeWidth={2}
                          dot={{ fill: colors.warning, strokeWidth: 2 }}
                          activeDot={{ r: 6 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="Medium"
                          name="Medium"
                          stroke={colors.secondary}
                          strokeWidth={2}
                          dot={{ fill: colors.secondary, strokeWidth: 2 }}
                          activeDot={{ r: 6 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="Low"
                          name="Low"
                          stroke={colors.success}
                          strokeWidth={2}
                          dot={{ fill: colors.success, strokeWidth: 2 }}
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
                                entry.severity === "High"
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
                        <Stack direction="row" spacing={2} alignItems="center">
                          <Chip
                            label={alert.severity.toUpperCase()}
                            sx={{
                              fontWeight: 600,
                              color: colors.textPrimary,
                              backgroundColor:
                                alert.severity === "High"
                                  ? "#B71C1C"
                                  : alert.severity === "Medium"
                                  ? "#F59E0B"
                                  : alert.severity === "Low"
                                  ? "#10B981"
                                  : alert.severity === "None"
                                  ? "#6B7280"
                                  : "#6B7280",
                              animation: newAlerts.some(
                                (a) => a.id === alert.id
                              )
                                ? "blink 1s ease-in-out 3"
                                : "none",
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

      {/* Notification for new alerts */}
      {openAlert && (
        <Box
          sx={{
            position: "fixed",
            top: 20,
            right: 20,
            zIndex: 9999,
            width: 350,
          }}
        >
          <Paper
            elevation={3}
            sx={{
              backgroundColor: colors.danger,
              color: colors.textPrimary,
              p: 2,
              borderRadius: "8px",
            }}
          >
            <Stack direction="row" alignItems="center" spacing={1}>
              <WarningIcon sx={{ fontSize: 30 }} />
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                  New High Threat Detected!
                </Typography>
                <Typography variant="body2">
                  {newAlerts.length} new{" "}
                  {newAlerts.length > 1 ? "alerts" : "alert"} requiring
                  immediate attention
                </Typography>
              </Box>
              <IconButton
                size="small"
                onClick={() => setOpenAlert(false)}
                sx={{ color: colors.textPrimary }}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            </Stack>
          </Paper>
        </Box>
      )}

      {/* CSS animation for blinking effect */}
      <style>{`
        @keyframes blink {
          0% { opacity: 1; }
          50% { opacity: 0.3; }
          100% { opacity: 1; }
        }
      `}</style>
    </Box>
  );
};

export default Dashboard;
