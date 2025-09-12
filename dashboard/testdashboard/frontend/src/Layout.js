import React, { useState, useEffect } from "react";
import { Outlet, Link } from "react-router-dom";
import {
  Box,
  Paper,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  useTheme,
  useMediaQuery,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import { Lock, Timeline } from "@mui/icons-material";
import { colors } from "./colors";

const Layout = () => {
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

  useEffect(() => {
    if (!isMobile && isMobileDrawerOpen) {
      setIsMobileDrawerOpen(false);
    }
  }, [isMobile, isMobileDrawerOpen]);

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
          "&:hover": { backgroundColor: colors.surface2 },
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
              component={Link}
              to="/"
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
              component={Link}
              to="/indexed-sites"
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

      {/* Main Content Area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          marginLeft: {
            xs: 0,
            md: isSidebarOpen ? "240px" : 0,
          },
          width: { md: `calc(100% - ${isSidebarOpen ? 240 : 0}px)` },
          position: "relative",
          py: 4,
          backgroundColor: colors.background,
          minHeight: "100vh",
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
};

export default Layout;