import {
  Card,
  CardActions,
  CardContent,
  CardHeader,
  Divider,
  IconButton,
  List,
  ListItem,
  ListItemText,
  TextField,
} from "@mui/material";
import { Send } from "@mui/icons-material";
import theme from "../theme";
import { createRef, useEffect, useState } from "react";

export type Message = {
  self: boolean;
  text: string;
};

export type ChatAreaProps = {
  messages: Message[];
  onMessage: (utterance: string) => void;
};

export default function ChatArea({ messages, onMessage }: ChatAreaProps) {
  const [text, setText] = useState("");
  const lastMessage = createRef<HTMLDivElement>();
  useEffect(() => {
    lastMessage.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  const handleSend = () => {
    if (text) {
      onMessage(text);
      setText("");
    }
  };
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setText(event.target.value);
  }
  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter'){
      handleSend();
    }
  }
  return (
    <Card
      variant="outlined"
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "50vh",
        position: "fixed",
        margin: 2,
      }}
    >
      <CardHeader title="Chat" subheader={<Divider />} />
      <CardContent sx={{ flex: 1, height: "100%", overflowY: "scroll" }}>
        <List sx={{}}>
          {messages.map((message, index) => (
            <ListItem key={index}>
              <ListItemText
                sx={{
                  textAlign: message.self ? "end" : "start",
                  background: message.self
                    ? theme.palette.secondary.main
                    : theme.palette.grey[200],
                  padding: 1,
                  borderRadius: 2,
                  borderTopLeftRadius: message.self ? undefined : 0,
                  borderTopRightRadius: message.self ? 0 : undefined,
                }}
                primary={message.text}
              ></ListItemText>
            </ListItem>
          ))}
          <div ref={lastMessage}></div>
        </List>
      </CardContent>
      <CardActions>
        <TextField
          fullWidth
          value={text}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          InputProps={{
            endAdornment: (
              <IconButton color="primary" onClick={handleSend}>
                <Send />
              </IconButton>
            ),
          }}
        />
      </CardActions>
    </Card>
  );
}
