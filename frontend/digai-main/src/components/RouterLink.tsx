import { Link as MuiLink, LinkProps } from "@mui/material";
import { Link } from "react-router-dom";

export default function RouterLink(props: LinkProps) {
  return <MuiLink component={Link} to={props.href ?? "#"} {...props} />;
}
