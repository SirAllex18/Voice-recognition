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
  CircularProgress
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
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  
  let mediaRecorder;
  let audioChunks = [];
  const startRecording = () => {
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

        // Stop recording after 3-5 seconds
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
    formData.append("audio", audioFile);

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
              style={{ marginTop: "20px" }}
            >
              {recording ? "Recording..." : "Start Recording"}
            </Button>
            {recording && <CircularProgress style={{ marginTop: "20px" }} />}
            {audioUrl && (
              <audio
                controls
                src={audioUrl}
                style={{ display: "block", marginTop: "20px" }}
              />
            )}
            {isUploading && <Typography>Uploading...</Typography>}
            {errorMessage && (
              <Typography color="error">{errorMessage}</Typography>
            )}
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
