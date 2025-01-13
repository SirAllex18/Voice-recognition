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
  CircularProgress,
} from "@mui/material";

const HomePage = () => {
  const [isLogin, setLogin] = useState(false);
  const [openDialog, setOpenDialog] = useState(false);
  const [formData, setFormData] = useState({
    nume: "",
    prenume: "",
    parola: "",
    mail: "",
  });
  const [recording, setRecording] = useState(false);
  const [alreadyRegistered, setRegistered] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [loggedInUser, setLoggedInUser] = useState(null);

  let mediaRecorder;
  let audioChunks = [];
  const startRecording = () => {
    if (isLogin === false) {
      setOpenDialog(true);
      return;
    }
    setRecording(true);
    setErrorMessage(null);

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();

        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
          const audioUrl = URL.createObjectURL(audioBlob);
          setAudioUrl(audioUrl);
          audioChunks = [];

          const audioFile = new File([audioBlob], "recording.wav", {
            type: "audio/wav",
          });

          uploadAudio(audioFile);
        };

        setTimeout(() => {
          mediaRecorder.stop();
          setRecording(false);
        }, 5000);
      })
      .catch((error) => {
        setErrorMessage("Error accessing microphone: " + error.message);
        setRecording(false);
      });
  };

  const uploadAudio = (audioFile) => {
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", audioFile);

    fetch("http://localhost:8000/recognize", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Upload successful:", data);
        setIsUploading(false);
      })
      .catch((error) => {
        console.error("Error uploading file:", error);
        setErrorMessage("Failed to upload audio.");
        setIsUploading(false);
      });
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({ ...prevData, [name]: value }));
  };

  const handleLoginButton = () => {
    setRegistered(true);
  };

  const handleLogout = () => {
    setLoggedInUser(null);
    setLogin(false);
    alert("You have been logged out");
  };

  const handleSubmitForm = async () => {
    if (!alreadyRegistered) {
      try {
        const response = await fetch("http://localhost:8000/register", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(formData),
        });

        const data = await response.json();

        if (response.ok) {
          alert("Registration successful!");
          setLoggedInUser(data);
          setOpenDialog(false);
          setLogin(true);
        } else {
          console.error("Error:", data.detail);
          alert(data.detail);
        }
      } catch (error) {
        console.error("Error during registration:", error);
        alert("Something went wrong!");
      }
    } else {
      try {
        const formDataLogin = new FormData();
        formDataLogin.append("email", formData.mail);
        formDataLogin.append("password", formData.password);

        const response = await fetch("http://localhost:8000/login", {
          method: "POST",
          body: formDataLogin,
        });

        const data = await response.json();

        if (response.ok) {
          alert("Login successful!");
          setLoggedInUser(data);
          localStorage.setItem("token", data.access_token);
          setOpenDialog(false);
          setLogin(true);
        } else {
          console.error("Error:", data.detail);
          alert(data.detail);
        }
      } catch (error) {
        console.error("Error during login:", error);
        alert("Something went wrong!");
      }
    }
  };
  console.log('Here is userdata', loggedInUser);

  return (
    <>
      <Typography variant="h4" align="center" marginTop="8rem">
        Your personal voice authentication application
      </Typography>
      {isLogin && (
        <Typography variant="body1" align="center">
          Hello, {loggedInUser.user.prenume}
      </Typography>
      )}
      
      <Container
        sx={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          height: "auto",
        }}
      >
        <Typography variant="body1" align="center" gutterBottom>
          Choose your usage:
        </Typography>
        <Box
          sx={{
            border: 2,
            borderRadius: 2,
            padding: 4,
            width: "300px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: 2,
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
              color="primary"
              onClick={startRecording}
              disabled={recording}
              style={{ width: "10rem" }}
            >
              {recording ? "Recording..." : "Authentication"}
            </Button>
            <Button
              variant="contained"
              color="primary"
              onClick={startRecording}
              disabled={recording}
              style={{ width: "10rem" }}
            >
              {recording ? "Recording..." : "Recognition"}
            </Button>
            {recording && <CircularProgress style={{ marginTop: "20px" }} />}
            {isUploading && <Typography>Uploading...</Typography>}
            {errorMessage && (
              <Typography color="error">{errorMessage}</Typography>
            )}
          </Box>
        </Box>
        {isLogin && (
          <Box>
            <Button onClick={handleLogout}> Logout </Button>
          </Box>
        )}
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
          {!alreadyRegistered && (
            <>
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
                label="E-mail"
                name="mail"
                value={formData.mail}
                onChange={handleInputChange}
              />
              <TextField
                label="Parola"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleInputChange}
              />
            </>
          )}

          {alreadyRegistered && (
            <>
              <TextField
                label="E-mail"
                name="mail"
                value={formData.mail}
                onChange={handleInputChange}
              />
              <TextField
                label="Parola"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleInputChange}
              />
            </>
          )}
        </DialogContent>
        <DialogActions
          sx={{
            display: "flex",
            flexDirection: "column",
            gap: 2,
          }}
        >
          <Box
            sx={{
              display: "flex",
              gap: 2,
              justifyContent: "center",
              width: "100%",
            }}
          >
            <Button onClick={handleCloseDialog}>Cancel</Button>
            <Button variant="contained" onClick={handleSubmitForm}>
              Submit
            </Button>
          </Box>
          <Button variant="text" onClick={handleLoginButton}>
            Already registered?
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default HomePage;
