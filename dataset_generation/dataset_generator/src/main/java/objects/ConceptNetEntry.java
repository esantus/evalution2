package objects;

public class ConceptNetEntry {
	
	String source;
	String relation;
	String target;
	String dataSource;
	String surfaceText;
	Double weight; 
	
	public ConceptNetEntry() {
		super();
		// TODO Auto-generated constructor stub
	}
	public ConceptNetEntry(String source, String relation, String target, String dataSource, String surfaceText, Double weight) {
		super();
		this.source = source;
		this.relation = relation;
		this.target = target;
		this.dataSource = dataSource;
		this.surfaceText = surfaceText;
		this.weight = weight;
	}
	public String getSource() {
		return source;
	}
	public void setSource(String source) {
		this.source = source;
	}
	public String getRelation() {
		return relation;
	}
	public void setRelation(String relation) {
		this.relation = relation;
	}
	public String getTarget() {
		return target;
	}
	public void setTarget(String target) {
		this.target = target;
	}
	public String getDataSource() {
		return dataSource;
	}
	public void setDataSource(String dataSource) {
		this.dataSource = dataSource;
	}
	public String getSurfaceText() {
		return surfaceText;
	}
	public void setSurfaceText(String surfaceText) {
		this.surfaceText = surfaceText;
	}
	public Double getWeight(){
		return weight;
	}
	public void setWeight(Double weight){
		this.weight = weight;
	}
	@Override
	public String toString() {
		return "ConceptNetEntry [source=" + source + ", relation=" + relation + ", target=" + target + ", dataSource="
				+ dataSource + ", surfaceText=" + surfaceText + ", weight="+ weight+"]";
	}
	
	@Override
	public int hashCode() {
	    return (getSource().hashCode() + getRelation().hashCode()+ getTarget().hashCode());
	}
	
	 @Override
	 public boolean equals (Object object) {
	     boolean result = false;
	     if (object == null || object.getClass() != getClass()) {
	         result = false;
	     } else {
	    	 ConceptNetEntry cn = (ConceptNetEntry) object;
	         if (this.source.equals(cn.getSource()) && this.relation.equals(cn.getRelation()) && this.target.equals(cn.getTarget())) {
	             result = true;
	         }
	         if(this.target.equals(cn.getSource()) & this.relation.equals(cn.getRelation()) && this.source.equals(cn.getTarget())){
	        	 result = true;
	         }
	     }
	     return result;
	 }
	
}

