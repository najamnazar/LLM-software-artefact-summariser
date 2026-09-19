package sumslicellm.messages;

/*
 * Derived from the released SumSlice source of P. W. McBurney and
 * C. McMillan (2013), bundled in this repository at
 *   ORIGINAL_SUMSLICE/sumslice/sumslice/messages/ReturnMessage.java.
 *
 * Unchanged from the released source apart from the package declaration.
 */

/**
 * A message representing the return type of a method.  For example, a
 * sentence generated for a message of this type might be "Method fooBar
 * returns an int."
 *
 * @author Collin McMillan
 * @since 2013-03-12
 */
public class ReturnMessage extends Message
{
	private String returntype;

	public String getReturnType()
	{
		return returntype;
	}

	public void setReturnType(String returntype)
	{
		this.returntype = returntype;
	}
}

