import { useNavigate } from "react-router-dom";
import { useState } from "react";
import {
  Container,
  Typography,
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from "@mui/material";

const HomePage = () => {
  const navigate = useNavigate();
  const [isLogin, setLogin] = useState(false);
  const [openDialog, setOpenDialog] = useState(false);
  const [formData, setFormData] = useState({
    nume: "",
    prenume: "",
    parola: "",
  });

  const handleAuthenticationLogic = () => {
    if (isLogin === false) {
      setOpenDialog(true);
    } else {
      console.log("Hello Bianca");
    }
  };

  const handleRecognitionLogic = () => {
    if (isLogin === false) {
      setOpenDialog(true); 
    } else {
      console.log("Hello Bianca");
    }
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({ ...prevData, [name]: value }));
  };

  const handleSubmitForm = () => {
    console.log("Form Submitted:", formData);
    setOpenDialog(false);
  };

  return (
    <>
      <Typography variant="h5" align="center" gutterBottom>
        Your personal Voice Authentication Application
      </Typography>
      <Container
        sx={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
        }}
      >
        <Typography variant="body2" align="center" gutterBottom>
          Choose your usage:
        </Typography>
        <Box
          sx={{
            border: 2,
            borderRadius: 2,
            padding: 4, 
            width: "300px", 
            textAlign: "center",
            boxShadow: 2
          }}
        >
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              gap: 1,
              alignItems: "center",
            }}
          >
            <Button
              variant="contained"
              sx={{
                width: "200px",
              }}
              onClick={handleAuthenticationLogic}
            >
              Authentication
            </Button>
            <Button
              variant="contained"
              sx={{
                width: "200px",
              }}
              onClick={handleRecognitionLogic}
            >
              Recognition
            </Button>
          </Box>
        </Box>
      </Container>

      {/* Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog}>
        <DialogTitle>Enter Your Details</DialogTitle>
        <DialogContent
          sx={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
            paddingTop: 2,
          }}
        >
          <TextField
            label="Nume"
            name="nume"
            value={formData.nume}
            onChange={handleInputChange}
          />
          <TextField
            label="Prenume"
            name="prenume"
            value={formData.prenume}
            onChange={handleInputChange}
          />
          <TextField
            label="Parola"
            name="parola"
            type="password"
            value={formData.parola}
            onChange={handleInputChange}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button variant="contained" onClick={handleSubmitForm}>
            Submit
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default HomePage;
