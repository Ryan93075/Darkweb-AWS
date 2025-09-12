// filepath: /d:/DarkEye/Dark-Web-Monitoring-main/DarkEye Dashboard/testdashboard/frontend/src/ProfessionalPaper.js
import { Paper } from "@mui/material";
import { styled } from "@mui/material/styles";
import { colors } from "./colors"; // Ensure colors are exported from this file

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

export default ProfessionalPaper;