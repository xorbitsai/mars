import React from "react";
import Title from "../Title";
import Paper from "@material-ui/core/Paper";

export default class NodeLogTab extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            loaded: false,
            content: null,
            interval: null,
            endpoint: null
        };
    }

    componentDidMount() {
        const intervalId = setInterval(() => this.loadLogs(), 10000);
        // store intervalId in the state, so it can be accessed later
        this.setState({
            interval: intervalId
        })
        this.loadLogs()
    }

    componentWillUnmount() {
        if (this.state.interval != null) {
            clearInterval(this.state.interval)
        }
    }

    render() {
        if (!this.state.loaded) {
            return (
                <div>Loading</div>
            );
        }
        return (
            <div>
                <Title component="h3">Generate Time: {new Date().toLocaleString()}</Title>
                <div>
                    <Paper style={{width: '100%', overflow: 'auto'}}>
                        <pre style={{fontSize: 'smaller'}}>{this.state.content}</pre>
                    </Paper>
                </div>
            </div>
        );
    }

    loadLogs() {
        fetch(`api/cluster/logs?address=${this.props.endpoint}`)
            .then((res) => res.json())
            .then((res) => {
                this.setState({
                    loaded: true,
                    content: res.content
                })
            })
    }
}
