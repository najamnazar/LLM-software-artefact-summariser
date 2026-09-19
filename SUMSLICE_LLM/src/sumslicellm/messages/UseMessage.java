package sumslicellm.messages;

/*
 * Derived from the released SumSlice source of P. W. McBurney and
 * C. McMillan (2013), bundled in this repository at
 *   ORIGINAL_SUMSLICE/sumslice/sumslice/messages/UseMessage.java.
 *
 * Unchanged from the released source apart from the package declaration.
 */

public class UseMessage extends Message {
	public String type;
	public String example;
	
	public void setType(String type){
		this.type = type;
	}
	
	public void setExample(String example){
		this.example = example;
	}
	
	public String getType(){
		return this.type;
	}
	
	public String getExample(){
		return this.example;
	}

}
