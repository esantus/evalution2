package net.codejava.hibernate;

import javax.persistence.*;

@Entity
@Table(name="allWordSenses")
public class AllWordSenses{
	@EmbeddedId
	private AllWordSensesID id;
	private Long source_id;
	
	public AllWordSenses() {
		super();
		// TODO Auto-generated constructor stub
	}

	public AllWordSenses(AllWordSensesID id) {
		super();
		this.id = id;
	}

	public AllWordSensesID getId() {
		return id;
	}

	public void setId(AllWordSensesID id) {
		this.id = id;
	}

	public Long getSource_id() {
		return source_id;
	}

	public void setSource_id(Long source_id) {
		this.source_id = source_id;
	}

	@Override
	public String toString() {
		return "AllWordSenses [id=" + id + ", source_id=" + source_id + "]";
	} 
	
	
	
}
