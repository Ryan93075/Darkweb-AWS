import React, { useState } from "react";
import {
  Container,
  Typography,
  Paper,
  Stack,
  Box,
  Chip,
  List,
  ListItem,
  ListItemText,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
} from "@mui/material";
import ProfessionalPaper from "./ProfessionalPaper";
import { colors } from "./colors";
import { BugReport, Warning, Error, Info, Lock, LocalPolice, LocalPharmacy, MoneyOff, Computer, People } from "@mui/icons-material";

const severityOptions = ["All", "High", "Medium", "Low", "None"];
const violationOptions = ["All", "firearms", "drugs", "fraud", "malware", "hacking", "human trafficking"];

const violationIcons = {
  malware: <BugReport fontSize="small" />,
  phishing: <Warning fontSize="small" />,
  exploit: <Error fontSize="small" />,
  darknet_market: <Lock fontSize="small" />,
  firearms: <LocalPolice fontSize="small" />,
  drugs: <LocalPharmacy fontSize="small" />,
  fraud: <MoneyOff fontSize="small" />,
  hacking: <Computer fontSize="small" />,
  'human trafficking': <People fontSize="small" />,
};

const IndexedSites = ({ data = [] }) => {
  const [selectedSeverity, setSelectedSeverity] = useState("All");
  const [selectedViolation, setSelectedViolation] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredData = data.filter((site) => {
    const matchesSeverity =
      selectedSeverity === "All" || site.severity === selectedSeverity;
    const matchesViolation =
      selectedViolation === "All" || site.violation === selectedViolation;
    const matchesSearch =
      site.onionsite_url &&
      site.onionsite_url.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSeverity && matchesViolation && matchesSearch;
  });

  const formatDate = (timestamp) => {
    if (!timestamp || isNaN(Date.parse(timestamp))) return "N/A";
    return new Date(timestamp).toLocaleString();
  };

  return (
    <Container maxWidth="xl">
      <Stack spacing={4}>
        <Stack spacing={1} sx={{ pl: 8 }}>
          <Typography
            variant="h4"
            sx={{
              color: colors.textPrimary,
              fontWeight: 700,
              letterSpacing: "-0.5px",
            }}
          >
            Indexed Onion Sites
          </Typography>
          <Typography
            variant="body1"
            sx={{
              color: colors.textSecondary,
              fontSize: "0.9rem",
            }}
          >
            Comprehensive List of Monitored Tor Sites
          </Typography>
        </Stack>

        <ProfessionalPaper>
          <Grid container spacing={3} alignItems="center" sx={{ mb: 3 }}>
            {/* Severity Filter */}
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel sx={{ color: colors.textSecondary }}>
                  Filter by Severity
                </InputLabel>
                <Select
                  value={selectedSeverity}
                  onChange={(e) => setSelectedSeverity(e.target.value)}
                  label="Filter by Severity"
                  sx={{
                    color: colors.textPrimary,
                    "& .MuiOutlinedInput-notchedOutline": {
                      borderColor: colors.textSecondary,
                    },
                    "& .MuiSelect-select": {
                      backgroundColor: colors.surface,
                    },
                  }}
                  MenuProps={{
                    PaperProps: {
                      sx: {
                        backgroundColor: "#1E293B", // Dark blue background
                        border: "1px solid #FFFFFF", // White border
                      },
                    },
                  }}
                >
                  {severityOptions.map((option) => (
                    <MenuItem key={option} value={option}>
                      <Chip
                        label={option}
                        sx={{
                          width: 80,
                          backgroundColor:
                            option === "High"
                              ? colors.danger
                              : option === "Medium"
                              ? colors.secondary
                              : option === "Low"
                              ? colors.success
                              : option === "None"
                              ? "#6B728055"
                              : "transparent",
                          color: colors.textPrimary,
                        }}
                      />
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            {/* Violation Filter */}
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel sx={{ color: colors.textSecondary }}>
                  Filter by Violation
                </InputLabel>
                <Select
                  value={selectedViolation}
                  onChange={(e) => setSelectedViolation(e.target.value)}
                  label="Filter by Violation"
                  sx={{
                    color: colors.textPrimary,
                    "& .MuiOutlinedInput-notchedOutline": {
                      borderColor: colors.textSecondary,
                    },
                    "& .MuiSelect-select": {
                      backgroundColor: colors.surface,
                    },
                  }}
                  MenuProps={{
                    PaperProps: {
                      sx: {
                        backgroundColor: "#1E293B", // Dark blue background
                        border: "1px solid #FFFFFF", // White border
                      },
                    },
                  }}
                >
                  {violationOptions.map((option) => (
                    <MenuItem key={option} value={option}>
                      <Stack direction="row" alignItems="center" spacing={1}>
                        {option !== "All" && (
                          violationIcons[option] || <Info fontSize="small" />
                        )}
                        <Typography sx={{ color: colors.textPrimary }}>
                          {option === "All" ? "All Violations" : option.charAt(0).toUpperCase() + option.slice(1)}
                        </Typography>
                      </Stack>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>

          <Typography variant="h6" sx={{ color: colors.textPrimary, mb: 2 }}>
            All Tracked Sites ({filteredData.length})
          </Typography>
          <Box sx={{ maxHeight: "600px", overflowY: "auto", pr: 1 }}>
            <List>
              {filteredData.map((site, index) => (
                <Paper
                  key={index}
                  sx={{
                    p: 2,
                    mb: 2,
                    borderRadius: "12px",
                    backgroundColor: colors.surface2,
                    borderLeft: `4px solid ${
                      site.severity === "High"
                        ? colors.danger
                        : site.severity === "Medium"
                        ? colors.secondary
                        : site.severity === "Low"
                        ? colors.success
                        : "#6B7280"
                    }`,
                  }}
                >
                  <ListItem disablePadding>
                    <Stack spacing={1} sx={{ width: "100%" }}>
                      <Grid container alignItems="center" spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Stack
                            direction="row"
                            spacing={2}
                            alignItems="center"
                          >
                            <Chip
                              label={site.severity.toUpperCase()}
                              sx={{
                                fontWeight: 600,
                                color: colors.textPrimary,
                                backgroundColor:
                                  site.severity === "High"
                                    ? colors.danger
                                    : site.severity === "Medium"
                                    ? colors.secondary
                                    : site.severity === "Low"
                                    ? colors.success
                                    : "#6B728055",
                                minWidth: 100,
                              }}
                            />
                            <Typography
                              variant="body1"
                              sx={{
                                color: colors.textPrimary,
                                wordBreak: "break-word",
                              }}
                            >
                              {site.onionsite_url}
                            </Typography>
                          </Stack>
                        </Grid>

                        <Grid item xs={12} md={6}>
                          <Stack spacing={1}>
                            {site.violation && (
                              <Stack
                                direction="row"
                                spacing={1}
                                alignItems="center"
                              >
                                <Typography
                                  variant="caption"
                                  sx={{
                                    color: colors.textSecondary,
                                    fontWeight: 1000,
                                  }}
                                >
                                  Violations:
                                </Typography>
                                <Stack
                                  direction="row"
                                  spacing={1}
                                  flexWrap="wrap"
                                >
                                  <Chip
                                    label={site.violation}
                                    size="medium"
                                    icon={
                                      violationIcons[site.violation] || (
                                        <Info fontSize="small" />
                                      )
                                    }
                                    sx={{
                                      backgroundColor: colors.surface,
                                      color: colors.textPrimary,
                                      "& .MuiChip-icon": {
                                        color: colors.textSecondary,
                                      },
                                    }}
                                  />
                                </Stack>
                              </Stack>
                            )}

                            <Typography
                              variant="caption"
                              sx={{
                                color:
                                  site.severity === "None"
                                    ? "#94A3B8"
                                    : colors.textSecondary,
                                fontStyle:
                                  formatDate(site.timestamp) === "N/A"
                                    ? "italic"
                                    : "normal",
                              }}
                            >
                              {`Severity: ${
                                site.severity
                              } | Detected: ${formatDate(site.timestamp)}`}
                            </Typography>
                          </Stack>
                        </Grid>
                      </Grid>

                      <ListItemText
                        primary={`ID: ${site._id.$oid}`}
                        sx={{
                          color: colors.textSecondary,
                          mt: 1,
                        }}
                      />
                    </Stack>
                  </ListItem>
                </Paper>
              ))}
            </List>
          </Box>
        </ProfessionalPaper>
      </Stack>
    </Container>
  );
};

export default IndexedSites;