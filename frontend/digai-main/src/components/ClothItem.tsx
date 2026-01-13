import {
  Card,
  CardActions,
  CardMedia,
  IconButton,
  ToggleButton,
} from "@mui/material";
import { Favorite, ShoppingCart } from "@mui/icons-material";
import React from "react";
export type ClothItemProps = {
  id: string;
  onLike: (liked: boolean) => void;
  onShop: (event: React.MouseEvent<HTMLButtonElement>) => void;
};
export default function ClothItem({ id, onLike, onShop }: ClothItemProps) {
  const [selected, setSelected] = React.useState(false);
  return (
    <Card>
      <CardMedia sx={{ aspectRatio: 1 }}
        image={`http://localhost:5001/images/${id}`}
      />
      <CardActions sx={{display: "flex", justifyContent: "space-between"}}>
        <ToggleButton
          color="primary"
          value="check"
          selected={selected}
          onChange={() => {
            const newState = !selected
            setSelected(newState);
            onLike(newState);
          }}
        >
          <Favorite />
        </ToggleButton>
        <IconButton onClick={onShop} color="primary"><ShoppingCart /></IconButton>
      </CardActions>
    </Card>
  );
}
