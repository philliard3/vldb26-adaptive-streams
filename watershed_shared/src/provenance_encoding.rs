// These are experiments in organizing provenance information.
// Provenance stores the ways in which data was derived
// - Why (which data a tuple originated from)
// - How (which operators we have gone through)
// - *Which* (routing decisions we made to go to the operators we did)

// this new "which" provenance is basically a routing-sensitive version of how provenance, as explained by why provenance

#[derive(Clone, Debug)]
struct PipelineId(u64);

#[repr(u8)]
enum ActionTag {
    Push = 0,
    Join = 1,
    Then = 2,
    // todo: information about a specific split decision?
    // we technically don't need this to reconstruct our execution tree
    // or to match the eventual execution tree with the end result
    // Split = 3,
}

#[derive(Clone, Debug)]
enum ExecutionTree {
    Operation {
        id: PipelineId,
    },
    // unary operations
    // todo: add unary operator specific data in a field
    Seq {
        prev: Box<ExecutionTree>,
        id: PipelineId,
    },
    Join {
        id: PipelineId,
        left: Box<ExecutionTree>,
        right: Box<ExecutionTree>,
    },
}

fn encode_execution1(e: &ExecutionTree) -> Vec<u8> {
    let mut output = Vec::new();
    encode_execution_recursive(e, &mut output);
    output
}

fn encode_execution_recursive(e: &ExecutionTree, output: &mut Vec<u8>) {
    match e {
        ExecutionTree::Operation { id } => {
            output.push(ActionTag::Push as _);
            output.extend(id.0.to_le_bytes());
        }
        ExecutionTree::Seq { prev, id } => {
            encode_execution_recursive(prev, output);
            output.push(ActionTag::Push as u8);
            output.extend(id.0.to_le_bytes());
            output.push(ActionTag::Then as u8);
        }
        ExecutionTree::Join { id, left, right } => {
            encode_execution_recursive(left, output);
            encode_execution_recursive(right, output);

            output.push(ActionTag::Join as u8);
            output.extend(id.0.to_le_bytes());
        }
    }
}

fn decode_execution1(mut src: &[u8]) -> Result<ExecutionTree, ()> {
    let mut tree_stack: Vec<ExecutionTree> = Vec::new();
    return decode_execution_recursive(&mut src);
    decode_execution_stack(&mut tree_stack, &mut src);
    match tree_stack.pop() {
        Some(v) => Ok(v),
        None => Err(()),
    }
}

// wait I'm actually not sure there's a way to make this recursive if it's already a stack machine
fn decode_execution_recursive(src: &mut &[u8]) -> Result<ExecutionTree, ()> {
    Err(())
}

// wait I'm actually not sure there's a way to make this recursive if it's already a stack machine
fn decode_execution_stack(tree_stack: &mut Vec<ExecutionTree>, src: &mut &[u8]) -> Result<(), ()> {
    Err(())
}
