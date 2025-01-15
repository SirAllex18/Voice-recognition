import { useState, useEffect } from "react";
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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
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
  const [registerFile, setRegisterFile] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [loggedInUser, setLoggedInUser] = useState(null);
  const [userVoiceRecording, setVoiceRecording] = useState(null);
  const [users, setUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isAdmin, setAdminPage] = useState(false);
  const [submitedVoice, setSubmitedVoice] = useState(false);
  let mediaRecorder;
  let audioChunks = [];

  useEffect(() => {
    const getUsers = async () => {
      try {
        const response = await fetch("http://localhost:8000/users");
        const initialData = await response.json();
        if (initialData.status === "success") {
          setUsers(initialData.users);
        }
      } catch (error) {
        console.error("Error fetching users:", error);
      } finally {
        setIsLoading(false);
      }
    };
    getUsers();
  }, [users]);

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
          setVoiceRecording(audioFile);
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
        console.log(data)
        alert(`Upload successful: ${data.status}\n Confidence: ${data.confidence}`);
        setIsUploading(false);
      })
      .catch((error) => {
        console.error("Error uploading file:", error);
        setErrorMessage("Failed to upload audio.");
        setIsUploading(false);
      });
  };

  const startRecordingAuthenticate = () => {
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

          uploadAudioAuthenticate(audioFile);
          setVoiceRecording(audioFile);
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

  const uploadAudioAuthenticate = (audioFile) => {
    setIsUploading(true);
    const fullName = loggedInUser.user.nume + loggedInUser.user.prenume;
    const formData = new FormData();
    formData.append("file", audioFile);
    formData.append("claimed_id", fullName);

    fetch("http://localhost:8000/authenticate", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        setIsUploading(false);
        if (data.status === "fail" && data.message === "Unknown speaker") {
          setRegisterFile(true);
          alert("No match found. If you wish, upload your voice to the dataset.")
        }
        if(data.status === "fail"){
          alert("Voice did not match claimed id. Authentication failed.")
        }
        if(data.status === "success"){
          alert("Authentication completed!")
        }  
        
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
    setAdminPage(false);
    setVoiceRecording(null);
    setSubmitedVoice(false);
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

  const handleSubmitDataset = async () => {
    const fullName = loggedInUser.user.nume + loggedInUser.user.prenume;
    const formData = new FormData();
    formData.append("user_id", fullName);
    formData.append("file", userVoiceRecording);
    formData.append("email", loggedInUser.user.mail);
    const submitDataset = await fetch("http://localhost:8000/enroll_voice", {
      method: "POST",
      body: formData,
    });
    const data = await submitDataset.json();
    if (submitDataset.ok) {
      alert("File uploaded succesfully")
      setSubmitedVoice(true)
      console.log("File uploaded succesfully");
    } else {
      console.log(data, "File not uploaded");
    }
  };

  const handleDeleteUser = async (userId) => {
    try {
      console.log(userId);
      const response = await fetch(
        `http://localhost:8000/deleteUserVoice/${userId}`,
        {
          method: "DELETE",
        }
      );
      const data = await response.json();

      if (response.ok) {
        setUsers((prevUsers) =>
          prevUsers.filter((user) => user._id !== userId)
        );
        alert(data.message);
      } else {
        console.error("Error deleting user:", data.detail);
        alert(data.detail);
      }
    } catch (error) {
      console.error("Error deleting user:", error);
      alert("Something went wrong while deleting the user.");
    }
  };
  if (loggedInUser?.user?.nume === "admin") {
    setAdminPage(true);
  }
  return (
    <>
      <Box
        sx={{
          height: "100vh",
          width: "100vw",
          backgroundColor: "rgba(9, 9, 11, 1)",
          color: "white",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <Typography variant="h4" align="center" marginBottom="4rem">
          Your personal voice authentication application
        </Typography>
        {isLogin && (
          <Typography variant="h6" align="center" marginBottom="0.15rem">
            Hello, {loggedInUser.user.prenume}
          </Typography>
        )}
        <Typography variant="body1" marginBottom="0.5rem">
          {" "}
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
              onClick={startRecordingAuthenticate}
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

        {registerFile && (
          <Box>
            <Button disabled={submitedVoice} onClick={handleSubmitDataset}>
              Submit voice into dataset
            </Button>
          </Box>
        )}

        {isLogin && (
          <Box>
            <Button onClick={handleLogout}>Logout</Button>
          </Box>
        )}

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
        {isAdmin && (
          <Container sx={{ marginTop: "2rem" }}>
            <Typography variant="h5" gutterBottom>
              User`s List
            </Typography>
            {isLoading ? (
              <CircularProgress />
            ) : (
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>#</TableCell>
                      <TableCell>User</TableCell>
                      <TableCell>Mail</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {users.map((user, index) => (
                      <TableRow key={user._id}>
                        <TableCell>{index + 1}</TableCell>
                        <TableCell>
                          {user.nume} {user.prenume}
                        </TableCell>
                        <TableCell>{user.mail}</TableCell>
                        <TableCell>
                          <Button
                            variant="contained"
                            color="secondary"
                            onClick={() => handleDeleteUser(user._id)}
                          >
                            Delete
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Container>
        )}
      </Box>
    </>
  );
};

export default HomePage;
